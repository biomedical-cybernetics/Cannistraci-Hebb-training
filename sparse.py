import torch
import torch.nn as nn
import math
import numpy as np
from scipy.io import savemat
import os
import matlab
import matplotlib.pyplot as plt



def find_first_pos(array, value):
    idx = (torch.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (torch.abs(array - value))
    idx = torch.flip(idx, dims=[0]).argmin()
    return array.shape[0] - idx
    


class sparse_layer(nn.Module):
    def __init__(self, indim, outdim, save_path, Tend, layer, eng, device, args):
        super(sparse_layer, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.remove_method = args.remove_method
        self.sparsity = args.sparsity
        
        self.zeta = args.zeta
        self.regrow_method = args.regrow_method
        self.save_path = save_path
        self.adaptive_zeta = args.adaptive_zeta
        self.Tend = Tend
        self.device = device
        self.eng = eng
        self.layer = layer
        self.early_stop = args.early_stop
        self.stop_signal = False
        self.early_stop_threshold = args.early_stop_threshold

        
        self.weight_mask = torch.rand(self.indim, self.outdim).to(self.device)
        self.weight_mask[self.weight_mask < self.sparsity] = 0
        self.weight_mask[self.weight_mask != 0] = 1
        self.n_params = len(self.weight_mask.nonzero())
        print('numbers of weights ', self.n_params)
        self.selected_model = []
        self.init_mode = args.init_mode
        self.update_mode = args.update_mode
        if self.init_mode == 'kaiming':
            self.stdv = math.sqrt(2 / self.indim)
        elif self.init_mode == 'xavier':
            self.stdv = math.sqrt(1 / self.indim)
        elif self.init_mode == 'gaussian':
            self.stdv = 1
        elif self.init_mode == 'swi':
            self.stdv = math.sqrt(2/(self.n_params / self.outdim))
            
        
        
        self.weight = nn.Parameter(torch.Tensor(self.indim, self.outdim))
        self.weight.data = (torch.randn(self.indim, self.outdim) * self.stdv).to(self.device)
        
        if args.bias:
            self.bias = nn.Parameter(torch.Tensor(self.outdim))
            self.bias.data = torch.zeros(self.outdim)
        else:
            self.register_parameter('bias', None)

        
        if self.bias is not None:
            self.bias.data = (torch.randn(self.outdim) * self.stdv).to(self.device)

        self.optimizer = None
        self.epoch = 0

        self.print_network = args.print_network
        if self.print_network:
            self.adjacency_save_path = self.save_path + "adj_" + str(self.layer) + "/"
            if not os.path.exists(self.adjacency_save_path):
                os.mkdir(self.adjacency_save_path)
            savemat(self.adjacency_save_path + str(self.epoch) + '.mat',
                    {"adjacency_matrix": self.weight_mask.cpu().numpy()})
        self.weight.data *= self.weight_mask

        self.overlap_rate = 0
        self.early_stop_signal = False
        
        
    @torch.no_grad
    def remove_connections(self):
        """
            Remove links
        """
        print('Number of weights before removal: ', torch.sum(self.weight_mask).item())
        
        self.mask_after_removal = torch.zeros_like(self.weight_mask)
        
        # if using adaptive zeta
        if self.adaptive_zeta:
            zeta = (float(self.zeta / 2) * (1 + math.cos(self.epoch * math.pi / self.Tend)))
            print("zeta: " + str(zeta))
        else:
            zeta = self.zeta
        
        
        
        if self.remove_method == "weight_magnitude":
            values = torch.sort((self.weight.data * self.weight_mask).ravel())[0]
            # remove connections
            firstZeroPos = find_first_pos(values, 0)  # Find the first_zero's index
            lastZeroPos = find_last_pos(values, 0)  # Find the last_zero's index
            self.largestNegative = values[int((1 - zeta) * firstZeroPos)]
            self.smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]

            print("smallest positive threshold: ", self.smallestPositive.item())
            print("largest negative threshold: ", self.largestNegative.item())

            rewiredWeights = self.weight.data * self.weight_mask
            rewiredWeights[rewiredWeights > self.smallestPositive] = 1
            rewiredWeights[rewiredWeights < self.largestNegative] = 1
            rewiredWeights[rewiredWeights != 1] = 0
        
        self.mask_after_removal = rewiredWeights
        print("Number of removal weights: ", int(torch.sum(self.weight_mask).item() - torch.sum(self.mask_after_removal).item()))
        
    @torch.no_grad
    def regrow_connections(self):
        """
            Regrow new links
        """
        self.noRewires = int(self.n_params - torch.sum(self.mask_after_removal).item())
        print("Number of regrown weights: ", self.noRewires)
        new_links_mask = torch.zeros((self.indim, self.outdim)).to(self.device)
        if self.regrow_method == "CH3_L3":    
            matlab_array = matlab.double(self.mask_after_removal.cpu().numpy().tolist())
            scores = np.array(self.eng.link_predict(matlab_array))
            thre = np.sort(scores.ravel())[-self.noRewires]
            if thre == 0:
                print("Regrowing threshold is 0!!!")
                thre = 0.00001
                pass
            # compute for missing links
            
            
            new_links_mask[scores >= thre] = 1
            new_links_mask[scores < thre] = 0

        elif self.regrow_method == "random":
            nrAdd = 0
            while (nrAdd < self.noRewires):
                i = np.random.randint(0, new_links_mask.shape[0])
                j = np.random.randint(0, new_links_mask.shape[1])
                if (new_links_mask[i, j] == 0 and self.mask_after_removal[i, j] == 0):
                    new_links_mask[i, j] = 1
                    nrAdd += 1

                
        
        # Assign value for new links
        if self.update_mode in ["kaiming", "xavier", "gaussian"]:
            value = (torch.randn(self.indim, self.outdim) * self.stdv).to(self.device)
            new_links_weight = new_links_mask * value
            
        elif self.update_mode == "swi":
            self.stdv =  (2/(self.n_params/self.num_output_active_nodes))
            value = (torch.randn(self.indim, self.outdim).to(self.device) * self.stdv)
            new_links_weight = new_links_mask * value
        elif self.update_mode == "zero":
            new_links_weight = torch.zeros(self.indim, self.outdim).to(self.device)
            
        
        
        # update weights
        self.weight.data *= (self.mask_after_removal + (new_links_mask * self.weight_mask))
        self.weight.data += (new_links_weight * (self.weight_mask == 0))
        removed_links_mask = self.weight_mask - self.mask_after_removal
        # update mask    
        self.weight_mask = self.mask_after_removal + new_links_mask
        self.new_links_mask = new_links_mask
        print('Number of weights after evolution: ', torch.sum(self.weight_mask).item())

        if self.print_network:
            savemat(self.adjacency_save_path + str(self.epoch) + '.mat',
                {"adjacency_matrix": self.weight_mask.cpu().numpy()})
            
        # Using early stop or not
        if self.early_stop:
            self.overlap_rate = torch.sum((removed_links_mask * new_links_mask) == 1) / self.noRewires
            print("Overlap rate of removal and regrown links are: ", self.overlap_rate.item())
            
            if self.overlap_rate > self.early_stop_threshold:
                print("The evolution of topology has stopped!")
                self.early_stop_signal = True

    @torch.no_grad
    def clear_buffers(self):
        """
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """

        removed_indics = self.weight_mask == 0
        buffers = list(self.optimizer.state[self.weight])
        for buffer in buffers:
            if buffer == 'momentum_buffer':
                # self.optimizer.state[self.weight]['momentum_buffer'] *= self.weight_mask
                self.optimizer.state[self.weight]['momentum_buffer'] *=(self.mask_after_removal + (self.new_links_mask * self.weight_mask))
            elif buffer == 'step_size':
                #set to learning rate
                print(len(self.optimizer.state[self.weight]['step_size'].nonzero()[0]))
                vals = removed_indics * self.optimizer.defaults['lr']
                self.optimizer.state[self.weight]['step_size'] = self.optimizer.state[self.weight]['step_size'] * self.weight_mask + vals
            elif buffer == 'prev':
                self.optimizer.state[self.weight]['prev'] *= self.weight_mask
            elif buffer == 'square_avg' or \
                buffer == 'exp_avg' or \
                buffer == 'exp_avg_sq' or \
                buffer == 'exp_inf':
                #use average of all for very rough estimate of stable value
                vals = self.weight_mask * torch.mean(self.optimizer.state[self.weight][buffer])
                self.optimizer.state[buffer] = vals
            elif buffer == 'acc_delta':
                #set to learning rate
                self.optimizer.state[self.weight]['step_size'] = self.weight_mask * zeros
            #elif buffer == ''

    def forward(self, x):
        self.weight_core = self.weight * self.weight_mask       
        x = torch.mm(x, self.weight_core)
            
        if self.bias is not None:
            x += self.bias
        
        return x

    
    def reset_parameters(self):
        self.weight.data = (torch.randn(self.indim, self.outdim) * self.stdv).to(self.device)
        self.weight.data *= self.weight_mask