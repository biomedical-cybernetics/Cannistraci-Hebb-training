import numpy as np
import torch
import torch.distributed as dist

from dst_util import get_W
from sparse_topology_initialization import create_ws_sparse_scheduler
import math
from scipy.sparse import csr_matrix
import sys
sys.path.append("../")
import CH_scores

def chain_removal(layer1, layer2):
    layer1 = remove_unactive_links_backward(layer1, layer2)
    layer2 = remove_unactive_links_forward(layer2, layer1)

    return layer1, layer2

def qk_chain_removal(q, k):
    q = remove_unactive_links_backward(q, k.transpose(1, 0))
    k = remove_unactive_links_backward(k, q.transpose(1, 0))

    return q, k


def remove_unactive_links_backward(current_adj, after_adj):
    outdegree = torch.sum(after_adj, dim=0)
    outdegree[outdegree>0] = 1
    current_num = torch.sum(current_adj)
    current_adj = current_adj * outdegree.reshape(-1, 1)
    print("Number of removed unactive links backwards: ", int(current_num - torch.sum(current_adj)))

    return current_adj

def remove_unactive_links_forward(current_adj, before_adj):
    indegree = torch.sum(before_adj, dim=1)
    indegree[indegree>0] = 1
    current_num = torch.sum(current_adj)
    current_adj = current_adj * indegree.reshape(1, -1)
    print("Number of removed unactive links forwards: ", int(current_num - torch.sum(current_adj)))
    return current_adj




class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class DSTScheduler:

    def __init__(self, model, optimizer, dense_allocation=1, T_end=None, sparsity_distribution='uniform', ignore_linear_layers=True, delta=100, alpha=0.3, static_topo=False, grad_accumulation_n=1, state_dict=None, args=None):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise Exception('Dense allocation must be on the interval (0, 1]. Got: %f' % dense_allocation)

        self.model = model
        self.optimizer = optimizer

        self.W, self._linear_layers_mask, self.chain_list, self.qk_chain_list = get_W(model, return_linear_layers_mask=True)

        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)
            
        self.dense_allocation = dense_allocation
        self.N = [torch.numel(w) for w in self.W]
        self.args = args
        if self.args.early_stop:
            self.early_stop_signal = torch.zeros(len(self.W))
        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()

        else:
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.grad_accumulation_n = grad_accumulation_n
            self.ignore_linear_layers = ignore_linear_layers
            self.backward_masks = None

            # define sparsity allocation
            self.S = []
            for i, (W, is_linear) in enumerate(zip(self.W, self._linear_layers_mask)):
                if self.args.EM_S:
                    self.S.append((1 - dense_allocation - 0.05))
                else:
                    self.S.append(1 - dense_allocation)
            
            if args.init_mode == "swi" or args.init_mode == "kaiming":
                # reset the parameters with swi
                self.reset_parameters()

            # randomly sparsify model according to S
            self.random_sparsify()

            # scheduler keeps a log of how many times it's called. this is how it does its scheduling
            self.step = 0
            self.dst_steps = 0

            # define the actual schedule
            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end

        # also, register backward hook so sparse elements cannot be recovered during normal training
        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, '_has_rigl_backward_hook', False):
                print(i, w.shape)
                # print()
                raise Exception('This model already has been registered to a DSTScheduler.')
        
            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])
            setattr(w, '_has_rigl_backward_hook', True)

        # assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta
        assert self.sparsity_distribution in ('uniform', )




    def state_dict(self):
        obj = {
            'dense_allocation': self.dense_allocation,
            'S': self.S,
            'N': self.N,
            'hyperparams': {
                'delta_T': self.delta_T,
                'alpha': self.alpha,
                'T_end': self.T_end,
                'ignore_linear_layers': self.ignore_linear_layers,
                'static_topo': self.static_topo,
                'sparsity_distribution': self.sparsity_distribution,
                'grad_accumulation_n': self.grad_accumulation_n,
            },
            'step': self.step,
            'dst_steps': self.dst_steps,
            'backward_masks': self.backward_masks,
            '_linear_layers_mask': self._linear_layers_mask,
        }

        return obj

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if type(v) == dict:
                self.load_state_dict(v)
            setattr(self, k, v)


    @torch.no_grad()
    def random_sparsify(self):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        self.record_mask = [] 
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue
            
            if self.args.WS:
                mask = create_ws_sparse_scheduler(self.S[l], w, self.args).contiguous()

            else:
                n = self.N[l]
                s = int(self.S[l] * n)
                perm = torch.randperm(n)
                perm = perm[:s]
                flat_mask = torch.ones(n, device=w.device)
                flat_mask[perm] = 0
                mask = torch.reshape(flat_mask, w.shape)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)
            if self.args.itop:
                self.record_mask.append(mask)


    def __str__(self):
        s = 'DSTScheduler(\n'
        s += 'layers=%i,\n' % len(self.N)

        # calculate the number of non-zero elements out of the total number of elements
        N_str = '['
        S_str = '['
        sparsity_percentages = []
        total_params = 0
        total_nonzero = 0

        for N, S, mask, W, is_linear in zip(self.N, self.S, self.backward_masks, self.W, self._linear_layers_mask):
            actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += ('%i/%i, ' % (N-actual_S, N))
            sp_p = float(N-actual_S) / float(N) * 100
            S_str += '%.2f%%, ' % sp_p
            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N-actual_S

        N_str = N_str[:-2] + ']'
        S_str = S_str[:-2] + ']'
        
        s += 'nonzero_params=' + N_str + ',\n'
        s += 'nonzero_percentages=' + S_str + ',\n'
        s += 'total_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_nonzero, total_params, float(total_nonzero)/float(total_params)*100)) + ',\n'
        # s += 'total_CONV_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_conv_nonzero, total_conv_params, float(total_conv_nonzero)/float(total_conv_params)*100)) + ',\n'
        s += 'step=' + str(self.step) + ',\n'
        s += 'num_dst_steps=' + str(self.dst_steps) + ',\n'
        s += 'ignoring_linear_layers=' + str(self.ignore_linear_layers) + ',\n'
        s += 'sparsity_distribution=' + str(self.sparsity_distribution) + ',\n'

        return s + ')'


    @torch.no_grad()
    def reset_momentum(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            param_state = self.optimizer._optimizer.state[w]
            optimizer_state_list = ["exp_avg", "exp_avg_sq"]
            for optimizer_state in optimizer_state_list:
                if optimizer_state in param_state:
                    # mask the momentum matrix
                    buf = param_state[optimizer_state]
                    buf *= mask

    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue
                
            w *= mask


    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            w.grad *= mask

    
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next rigl step is, 
        if it's within `self.grad_accumulation_n` steps, return True.
        """

        if self.step >= self.T_end:
            return False

        steps_til_next_rigl_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_rigl_step <= self.grad_accumulation_n


    def cosine_annealing(self):
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))


    def __call__(self):
        self.step += 1
        if self.static_topo:
            return True
        if self.args.early_stop:
            if torch.sum(self.early_stop_signal) == len(self.W):
                # print("All layer early stopped!")
                return True
        if (self.step % self.delta_T) == 0 and self.step < self.T_end: # check schedule
            
            self._dst_step()
            self.dst_steps += 1
            print(self)
            return False
        return True


    @torch.no_grad()
    def _dst_step(self):
        if self.args.adaptive_zeta:
            drop_fraction = self.cosine_annealing()
        else:
            drop_fraction = self.alpha

        # if distributed these values will be populated
        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        if self.args.chain_removal:
            # If use chain_removal, has to divide the evolution of the mask into two steps
            # only for chts: CH3_L3njp_soft, CH3_L3njp_soft2, CH3_L3_soft, CH3_L3
            mask1_total = []
            n_prune = []
            n_keep = []
            n_ones = []
            for l, w in enumerate(self.W):
                # Link removal
                # if sparsity is 0%, skip
                if self.args.EM_S:
                    drop_fraction = (1-self.S[l]-self.dense_allocation)/(1-self.S[l])
                
                if self.S[l] <= 0:
                    continue

                current_mask = self.backward_masks[l]
                

                # calculate drop/grow quantities
                n_total = self.N[l]
                n_ones.append(torch.sum(current_mask).item())
                n_prune.append(int(n_ones[l] * drop_fraction))
                n_keep.append(n_ones[l] - n_prune[-1])

                # create drop mask
                if self.args.remove_method == "weight_magnitude":
                    # calculate raw scores
                    score_drop = torch.abs(w)
                    # if is distributed, synchronize scores
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)

                    _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                    new_values = torch.where(
                                torch.arange(n_total, device=w.device) < n_keep[-1],
                                torch.ones_like(sorted_indices),
                                torch.zeros_like(sorted_indices))
                    mask1 = new_values.scatter(0, sorted_indices, new_values)



                elif self.args.remove_method == "weight_magnitude_soft":
                    score_drop = torch.abs(w)
                    # if is distributed, synchronize scores
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)
                    T = 1 + self.step * (2 / self.T_end)
                    # print(f"Current Temperature: {T}")

                    mask1 = torch.zeros_like(score_drop.view(-1)).to(w.device)
                    flat_matrix = (score_drop.flatten())** T
                    probabilities = flat_matrix / flat_matrix.sum()

                    sampled_flat_indices = torch.multinomial(probabilities, max(1, n_keep[-1]), replacement=False)
                    mask1[sampled_flat_indices] = 1
                
                elif self.args.remove_method == "ri":
                    score_drop = torch.abs(w)
                    # if is distributed, synchronize scores
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)
                    eplison = 0.00001
                    score_drop = torch.abs(score_drop)/torch.sum(torch.abs(score_drop)+ eplison, dim=0) + torch.abs(score_drop)/torch.sum(torch.abs(score_drop)+ eplison, dim=1).reshape(-1, 1)
                    _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                    new_values = torch.where(
                                torch.arange(n_total, device=w.device) < n_keep[-1],
                                torch.ones_like(sorted_indices),
                                torch.zeros_like(sorted_indices))
                    mask1 = new_values.scatter(0, sorted_indices, new_values)

                elif self.args.remove_method == "ri_soft":
                    eplison = 0.00001
                    score_drop = torch.abs(w)
                    # if is distributed, synchronize scores
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)
                    score_drop = torch.abs(score_drop)/torch.sum(torch.abs(score_drop)+ eplison, dim=0) + torch.abs(score_drop)/torch.sum(torch.abs(score_drop)+ eplison, dim=1).reshape(-1, 1)
                    T = 1 + self.step * (2 / self.T_end)
                    # print(f"Current Temperature: {T}")

                    mask1 = torch.zeros_like(score_drop.view(-1)).to(w.device)
                    flat_matrix = (score_drop.flatten())** T
                    probabilities = flat_matrix / flat_matrix.sum()

                    sampled_flat_indices = torch.multinomial(probabilities, max(1, n_keep[-1]), replacement=False)
                    mask1[sampled_flat_indices] = 1
                    
                else:
                    raise NotImplementedError
                
                # print("Number of removal: ", n_prune[l])
                mask1_total.append(torch.reshape(mask1, current_mask.shape))

            # Chain removal
            for chain in self.qk_chain_list:
                mask1_total[chain[0]], mask1_total[chain[1]] = qk_chain_removal(mask1_total[chain[0]], mask1_total[chain[1]])

            for chain in self.chain_list:
                mask1_total[chain[0]], mask1_total[chain[1]] = chain_removal(mask1_total[chain[0]], mask1_total[chain[1]])



            for l, w in enumerate(self.W):
                current_mask = self.backward_masks[l]
                if self.args.EM_S:
                    if self.step <= self.T_end * 0.6:
                        self.S[l] = 1-self.dense_allocation-0.05
                        n_prune[l] = int(0.05 * self.N[l])
                    elif self.step < (self.T_end - self.delta_T):
                        self.S[l] = 1-self.dense_allocation-0.025
                        n_prune[l] = int(0.025 * self.N[l])
                    else:
                        self.S[l] = 1-self.dense_allocation
                        n_prune[l] = 0
                        current_mask.data = torch.reshape(mask1_total[l], current_mask.shape)
                        print("Final sparsity: {}".format(torch.sum(mask1_total[l]).item()/self.N[l]))
                        # exit()
                        
                        # print(torch.sum(mask_combined, dim=0)[:100])
                        self.reset_momentum()
                        self.apply_mask_to_weights()
                        self.apply_mask_to_gradients() 
                        continue
                else:
                    n_prune[l] = int(n_ones[l] - torch.sum(mask1_total[l]))
                    # print(f"Sparse layer {l}, number of regrowth links: {n_prune[l]}")
                

                # Link regrowth
                CH_method = self.args.regrow_method.split("_")[0]

                if "L3n" in self.args.regrow_method:
                    
                    DTPATHS1 = mask1_total[l].clone().float()
                    
                    TDPATHS1 = DTPATHS1.transpose(1, 0)
                    DDPATHS2 = torch.matmul(DTPATHS1, TDPATHS1)
                    TTPATHS2 = torch.matmul(TDPATHS1, DTPATHS1)

                    BDDPATHS2 = DDPATHS2 != 0
                    BTTPATHS2 = TTPATHS2 != 0

                    elcl_DT = (torch.sum(DTPATHS1, dim=1) - DDPATHS2) * BDDPATHS2
                    elcl_TD = (torch.sum(TDPATHS1, dim=1) - TTPATHS2) * BTTPATHS2

                    elcl_DT[elcl_DT == 0] = 1
                    elcl_TD[elcl_TD == 0] = 1

                    elcl_DT -= 1
                    elcl_TD -= 1
                    if CH_method == "CH2":
                        elcl_DT = 1 / (elcl_DT + 1) * (DDPATHS2 + BDDPATHS2)
                        elcl_TD = 1 / (elcl_TD + 1) * (TTPATHS2 + BTTPATHS2)
                    elif CH_method == "CH3":
                        elcl_DT = 1 / (elcl_DT + 1) * BDDPATHS2
                        elcl_TD = 1 / (elcl_TD + 1) * BTTPATHS2
                    elif CH_method == "CH3.1":
                        pass
                    

                    elcl_DT = torch.matmul(elcl_DT, DTPATHS1)
                    elcl_TD = torch.matmul(elcl_TD, TDPATHS1)

                    scores = elcl_DT + elcl_TD.T
                    scores = scores * (mask1_total[l] == 0)
                    thre = torch.sort(scores.ravel())[0][-n_prune[l]]
                    if thre == 0:
                        print("Regrowing threshold is 0!!!")
                        scores = (scores + 0.00001)*(mask1_total[l]==0)
                
                elif "L3p" in self.args.regrow_method:
                    # CH3_L3 path-based regrowth
                    xb = np.array(mask1_total[l].cpu())
                    x = transform_bi_to_mo(xb)
                    
                    A = csr_matrix(x)
                    ir = A.indices
                    jc = A.indptr
                    if CH_method == "CH2":
                        scores_cell = torch.tensor(np.array(CH_scores.CH_scores_new_v2(ir, jc, x.shape[0], [3], 1, 3, [2], 1))).to(w.device)
                    elif CH_method == "CH3":
                        scores_cell = torch.tensor(np.array(CH_scores.CH_scores_new_v2(ir, jc, x.shape[0], [3], 1, 3, [3], 1))).to(w.device)
                    elif CH_method == "CH3.1":
                        scores_cell = torch.tensor(np.array(CH_scores.CH_scores_new_v2(ir, jc, x.shape[0], [3], 1, 3, [5], 1))).to(w.device)
                    else:
                        raise NotImplementedError
                    scores = torch.reshape(scores_cell, x.shape)
                    scores = scores[:xb.shape[0], xb.shape[0]:]
                    
                    scores = scores * (mask1_total[l] == 0)

                    thre = torch.sort(scores.ravel())[0][-n_prune[l]]
                    if thre == 0:
                        print("Regrowing threshold is 0!!!")
                        print(f"# of scores: {torch.sum(scores > 0)}")
                        scores = (scores + 0.00001)*(mask1_total[l]==0)
                elif self.args.regrow_method == "random":
                    # random regrowth
                    scores = torch.rand(w.shape).to(w.device) * (mask1_total[l] == 0)
                    # flatten grow scores
                    thre = torch.sort(scores.ravel())[0][-n_prune[l]]

                elif self.args.regrow_method == "gradient":
                    scores = torch.abs(self.backward_hook_objects[l].dense_grad) * (mask1_total[l] == 0)
                    # flatten grow scores
                    thre = torch.sort(scores.ravel())[0][-n_prune[l]]
                    
                else:
                    raise NotImplementedError


                if "soft" in self.args.regrow_method:
                    mask2 = torch.zeros_like(scores.view(-1)).to(w.device)
                    flat_matrix = scores.flatten()
                    probabilities = flat_matrix / flat_matrix.sum()
                    # print(probabilities.shape)
                    sampled_flat_indices = torch.multinomial(probabilities, max(1, n_prune[l]), replacement=False)
                    mask2[sampled_flat_indices] = 1
                else:
                    mask2 = torch.zeros_like(scores).to(w.device)
                    mask2[scores >= thre] = 1
                if self.args.tiedrank:
                    pass

                mask2_reshaped = torch.reshape(mask2, current_mask.shape)
                grow_tensor = torch.zeros_like(w)
                

                if self.args.early_stop:
                    print("Overlap rate: ", (torch.sum((mask2_reshaped == 1) & (current_mask.int() - mask1_total[l])) / n_prune[l]).item())
                    if (torch.sum((mask2_reshaped == 1) & (current_mask.int() - mask1_total[l])) / n_prune[l]) > self.args.early_stop_thre:
                        self.early_stop_signal[l] = 1 
                
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

                # update new weights to be initialized as zeros and update the weight tensors
                new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
                w.data = new_weights

                mask_combined = (mask1_total[l] + mask2_reshaped).bool()
                if self.args.itop:
                    self.record_mask[l] = ((self.record_mask[l] == 1) | (mask_combined == 1))
                    print("ITOP rate is : ", (torch.sum(self.record_mask[l]) / mask_combined.numel()).item())

                # update the mask
                current_mask.data = mask_combined

                self.reset_momentum()
                self.apply_mask_to_weights()
                self.apply_mask_to_gradients() 



        else:
            # No chain removal
            for l, w in enumerate(self.W):
                # if sparsity is 0%, skip
                if self.S[l] <= 0:
                    continue
                if self.args.EM_S:
                    drop_fraction = (1-self.S[l]-self.dense_allocation)/(1-self.S[l])

                current_mask = self.backward_masks[l]

                # calculate drop/grow quantities
                n_total = self.N[l]
                n_ones = torch.sum(current_mask).item()
                n_prune = int(n_ones * drop_fraction)
                n_keep = n_ones - n_prune

                # create drop mask
                if self.args.remove_method == "weight_magnitude":
                    '''
                    weight magnitude based pruning
                    '''
                    score_drop = torch.abs(w)
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)
                    _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                    new_values = torch.where(
                                torch.arange(n_total, device=w.device) < n_keep,
                                torch.ones_like(sorted_indices),
                                torch.zeros_like(sorted_indices))
                    mask1 = new_values.scatter(0, sorted_indices, new_values)
                elif self.args.remove_method == "MEST":
                    score_drop = torch.abs(w)
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)
                    
                    score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)
                    if is_dist:

                        dist.all_reduce(score_grow)  # get the sum of all grow scores
                        score_grow /= world_size     # divide by world size (average the grow scores)

                    score_drop = score_drop + 0.01 * torch.abs(score_grow * current_mask)
                    _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                    new_values = torch.where(
                                torch.arange(n_total, device=w.device) < n_keep,
                                torch.ones_like(sorted_indices),
                                torch.zeros_like(sorted_indices))
                    mask1 = new_values.scatter(0, sorted_indices, new_values)
                elif self.args.remove_method == "weight_magnitude_soft":
                    score_drop = torch.abs(w)
                    # if is distributed, synchronize scores
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)
                    T = 1 + self.step * (2 / self.T_end)
                    # print(f"Current Temperature: {T}")

                    mask1 = torch.zeros_like(score_drop.view(-1)).to(w.device)
                    flat_matrix = (score_drop.flatten())** T
                    probabilities = flat_matrix / flat_matrix.sum()

                    sampled_flat_indices = torch.multinomial(probabilities, max(1, n_keep[-1]), replacement=False)
                    mask1[sampled_flat_indices] = 1
                
                elif self.args.remove_method == "ri":
                    score_drop = torch.abs(w)
                    # if is distributed, synchronize scores
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)
                    eplison = 0.00001
                    score_drop = torch.abs(score_drop)/torch.sum(torch.abs(score_drop)+ eplison, dim=0) + torch.abs(score_drop)/torch.sum(torch.abs(score_drop)+ eplison, dim=1).reshape(-1, 1)
                    _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                    new_values = torch.where(
                                torch.arange(n_total, device=w.device) < n_keep[-1],
                                torch.ones_like(sorted_indices),
                                torch.zeros_like(sorted_indices))
                    mask1 = new_values.scatter(0, sorted_indices, new_values)

                elif self.args.remove_method == "ri_soft":
                    eplison = 0.00001
                    score_drop = torch.abs(w)
                    # if is distributed, synchronize scores
                    if is_dist:
                        dist.all_reduce(score_drop)  # get the sum of all drop scores
                        score_drop /= world_size     # divide by world size (average the drop scores)
                    score_drop = torch.abs(score_drop)/torch.sum(torch.abs(score_drop)+ eplison, dim=0) + torch.abs(score_drop)/torch.sum(torch.abs(score_drop)+ eplison, dim=1).reshape(-1, 1)
                    T = 1 + self.step * (2 / self.T_end)
                    # print(f"Current Temperature: {T}")

                    mask1 = torch.zeros_like(score_drop.view(-1)).to(w.device)
                    flat_matrix = (score_drop.flatten())** T
                    probabilities = flat_matrix / flat_matrix.sum()

                    sampled_flat_indices = torch.multinomial(probabilities, max(1, n_keep[-1]), replacement=False)
                    mask1[sampled_flat_indices] = 1
                else:
                    raise NotImplementedError


                if self.args.EM_S:
                    if self.step <= self.T_end * 0.6:
                        self.S[l] = 1-self.dense_allocation-0.05
                        n_prune = int(0.05 * self.N[l])
                    elif self.step < (self.T_end - self.delta_T):
                        self.S[l] = 1-self.dense_allocation-0.025
                        n_prune = int(0.025 * self.N[l])
                    else:
                        self.S[l] = 1-self.dense_allocation
                        n_prune = 0
                        current_mask.data = torch.reshape(mask1, current_mask.shape)
                        
                        # print(torch.sum(mask_combined, dim=0)[:100])
                        self.reset_momentum()
                        self.apply_mask_to_weights()
                        self.apply_mask_to_gradients() 
                        continue


                if self.args.regrow_method == "gradient":
                    '''
                    gradient based regrowth
                    '''
                    score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)
                    if is_dist:

                        dist.all_reduce(score_grow)  # get the sum of all grow scores
                        score_grow /= world_size     # divide by world size (average the grow scores)
                    # flatten grow scores
                    score_grow = score_grow.view(-1)

                    # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
                    score_grow_lifted = torch.where(
                                        mask1 == 1, 
                                        torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                        score_grow)

                    # create grow mask
                    _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
                    new_values = torch.where(
                                    torch.arange(n_total, device=w.device) < n_prune,
                                    torch.ones_like(sorted_indices),
                                    torch.zeros_like(sorted_indices))
                    mask2 = new_values.scatter(0, sorted_indices, new_values)

                elif self.args.regrow_method == "random":
                    # random regrowth
                    score_grow = torch.rand(w.shape).to(w.device)
                    # flatten grow scores
                    score_grow = score_grow.view(-1)

                    # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
                    score_grow_lifted = torch.where(
                                        mask1 == 1, 
                                        torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                        score_grow)

                    # create grow mask
                    _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
                    new_values = torch.where(
                                    torch.arange(n_total, device=w.device) < n_prune,
                                    torch.ones_like(sorted_indices),
                                    torch.zeros_like(sorted_indices))
                    mask2 = new_values.scatter(0, sorted_indices, new_values)
                else:
                    raise NotImplementedError
                

                mask2_reshaped = torch.reshape(mask2, current_mask.shape)
                grow_tensor = torch.zeros_like(w)
                
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

                # update new weights to be initialized as zeros and update the weight tensors
                new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
                w.data = new_weights

                mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()

                if self.args.itop:
                    self.record_mask[l] = ((self.record_mask[l] == 1) | (mask_combined == 1))
                    print("ITOP rate is : ", (torch.sum(self.record_mask[l]) / mask_combined.numel()).item())

                # update the mask
                current_mask.data = mask_combined

                self.reset_momentum()
                self.apply_mask_to_weights()
                self.apply_mask_to_gradients() 
    @torch.no_grad()
    def reset_parameters(self):
        for l, w in enumerate(self.W):
            if self.args.init_mode == "swi":
                stdv = math.sqrt(2. / (((1-self.S[l]) * self.N[l]) / w.shape[1]))
            elif self.args.init_mode == "kaiming":
                stdv = math.sqrt(2 / w.shape[1])
            else:
                raise NotImplementedError
            w.data = (torch.randn(w.shape[0], w.shape[1]) * stdv).to(w.device)

def transform_bi_to_mo(xb):
    # create monopartite adjacency matrix
    x = np.zeros((xb.shape[0] + xb.shape[1], xb.shape[0] + xb.shape[1]))

    # Assign xb to the top-right block of matrix x
    x[:xb.shape[0], xb.shape[0]:] = xb

    # Assign the transpose of xb to the bottom-left block of matrix x
    x[xb.shape[0]:, :xb.shape[0]] = xb.T
    return x
