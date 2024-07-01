from sparse import sparse_layer
import torch.nn as nn
from scipy.io import savemat
import torch

def remove_unactive_links_backward(current_adj, after_adj):
    outdegree = torch.sum(after_adj, dim=1)
    outdegree[outdegree>0] = 1
    current_num = torch.sum(current_adj)
    current_adj = current_adj * outdegree

    print("Number of removed unactive links backwards: ", int(current_num - torch.sum(current_adj)))

    return current_adj

def remove_unactive_links_forward(current_adj, before_adj):
    indegree = torch.sum(before_adj, dim=0)
    indegree[indegree>0] = 1
    current_num = torch.sum(current_adj)
    current_adj = current_adj * indegree.reshape(-1, 1)

    print("Number of removed unactive links forwards: ", int(current_num - torch.sum(current_adj)))

    return current_adj

class sparse_mlp(nn.Module):
    def __init__(self, indim, hiddim, outdim, save_path, Tend, eng, device, args) -> None:
        super(sparse_mlp, self).__init__()
        self.sparse_layer1 = sparse_layer(indim, hiddim[0], save_path, Tend, 1, eng, device, args)
        self.sparse_layer2 = sparse_layer(hiddim[0], hiddim[1], save_path, Tend, 2, eng, device, args)
        self.sparse_layer3 = sparse_layer(hiddim[1], hiddim[2], save_path, Tend, 3, eng, device, args)
        self.sparse_layers = [self.sparse_layer1, self.sparse_layer2, self.sparse_layer3]
        
        self.last_layer  = nn.Linear(hiddim[2], outdim)
        self.chain_removal = args.chain_removal
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout)
        self.update_mode = args.update_mode
        self.clear_buffer = args.clear_buffer
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.dropout(self.relu(self.sparse_layer1(x.reshape(batch_size, -1))))
        out = self.relu(self.sparse_layer2(out))
        out = self.relu(self.sparse_layer3(out))
        out = self.last_layer(out)

        return out

    def evolve_connections(self):
        Flag = True
        for i, layer in enumerate(self.sparse_layers):
            if not self.sparse_layers[i].early_stop_signal:
                Flag = False
        
        if Flag:
            print("Early stop all the topological evolutions")
            return
            
        # remove connections
        for i, layer in enumerate(self.sparse_layers):
            layer.remove_connections()
        
        if self.update_mode == "swi":
            for i, layer in enumerate(self.sparse_layers):
                layer.num_output_active_nodes = torch.sum(torch.sum(layer.mask_after_removal, dim=0)!=0)

        
        # chain removal
        if self.chain_removal:
            for i in reversed(range(len(self.sparse_layers)-1)):
                self.sparse_layers[i].mask_after_removal = remove_unactive_links_backward(self.sparse_layers[i].mask_after_removal, self.sparse_layers[i+1].mask_after_removal)

            for i in range(1, len(self.sparse_layers)):
                self.sparse_layers[i].mask_after_removal = remove_unactive_links_forward(self.sparse_layers[i].mask_after_removal, self.sparse_layers[i-1].mask_after_removal)
            
        
        # regrow connections
        for i, layer in enumerate(self.sparse_layers):
            layer.regrow_connections()
            if self.clear_buffer:
                layer.clear_buffers()
            layer.epoch += 1

            if layer.print_network:
                savemat(layer.adjacency_save_path + str(layer.epoch) + '.mat',
                        {"adjacency_matrix": layer.weight_mask.cpu().numpy()})
        # exit()

class dense_mlp(nn.Module):
    def __init__(self, indim, hiddim, outdim, args) -> None:
        super(dense_mlp, self).__init__()
        self.Linear1 = nn.Linear(indim, hiddim[0])
        self.Linear2 = nn.Linear(hiddim[0], hiddim[1])
        self.Linear3 = nn.Linear(hiddim[1], hiddim[2])
        self.last_layer = nn.Linear(hiddim[2], outdim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.dropout(self.relu(self.Linear1(x.reshape(batch_size, -1))))
        out = self.dropout(self.relu(self.Linear2(out)))
        out = self.dropout(self.relu(self.Linear3(out)))
        out = self.last_layer(out)

        return out


class Dense_GoogleNet(nn.Module):
    def __init__(self, indim, hiddim, outdim):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout2d(p=0.4)
        # self.linear = nn.Linear(1024, num_class)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, hiddim)
        self.fc3 = nn.Linear(hiddim, hiddim)
        self.sparse_layers = [self.fc1, self.fc2, self.fc3]
        self.fc = nn.Linear(hiddim, outdim)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        output = self.relu(self.fc1(x))
        output = self.relu(self.fc2(output))
        output = self.relu(self.fc3(output))   
        x = self.fc(output)

        return x
    
class Sparse_GoogleNet(nn.Module):

    def __init__(self, indim, hiddim, outdim, save_path, Tend, eng, device, args):
        super(Sparse_GoogleNet, self).__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.chain_removal = args.chain_removal
        self.cnn_layers = nn.Sequential(self.prelayer, self.maxpool, self.a3, self.b3,
                                        self.maxpool, self.a4, self.b4, self.c4, self.d4, self.e4,
                                        self.maxpool, self.a5, self.b5, self.avgpool)
        
        self.relu = nn.ReLU(inplace=True)
        self.sl1 = sparse_layer(indim, hiddim, save_path, Tend, 1, eng, device, args)
        self.sl2 = sparse_layer(hiddim, hiddim, save_path, Tend, 2, eng, device, args)
        self.sl3 = sparse_layer(hiddim, hiddim, save_path, Tend, 3, eng, device, args)
        self.sparse_layers = [self.sl1, self.sl2, self.sl3]

        self.update_mode = args.update_mode
        self.fc = nn.Linear(hiddim, outdim)
        self.clear_buffer = args.clear_buffer

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        for i in self.sparse_layers:
            x = self.relu(i(x))
        x = self.fc(x)

        return x

    def evolve_connections(self):
        Flag = True
        for i, layer in enumerate(self.sparse_layers):
            if not self.sparse_layers[i].early_stop_signal:
                Flag = False
        
        if Flag:
            print("Early stop all the topological evolutions")
            return
            
        # remove connections
        for i, layer in enumerate(self.sparse_layers):
            layer.remove_connections()
            
        # chain removal
        if self.chain_removal:
            for i in reversed(range(len(self.sparse_layers)-1)):
                self.sparse_layers[i].mask_after_removal = remove_unactive_links_backward(self.sparse_layers[i].mask_after_removal, self.sparse_layers[i+1].mask_after_removal)

            for i in range(1, len(self.sparse_layers)):
                self.sparse_layers[i].mask_after_removal = remove_unactive_links_forward(self.sparse_layers[i].mask_after_removal, self.sparse_layers[i-1].mask_after_removal)
            
        if self.update_mode == "swi":
            for i, layer in enumerate(self.sparse_layers):
                layer.num_output_active_nodes = torch.sum(torch.sum(layer.mask_after_removal, dim=0)!=0)
        
        
        # regrow connections
        for i, layer in enumerate(self.sparse_layers):
            layer.regrow_connections()
            if self.clear_buffer:
                layer.clear_buffers()
            layer.epoch += 1

            if layer.print_network:
                savemat(layer.adjacency_save_path + str(layer.epoch) + '.mat',
                        {"adjacency_matrix": layer.weight_mask.cpu().numpy()})
    
    
class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

    
    
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Dense_ResNet152(nn.Module):

    def __init__(self, args, num_block, outdim=100):
        super().__init__()

        self.in_channels = 64
        num_block = [3, 8, 36, 3]
        block=BottleNeck
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 512 * block.expansion * args.dim)
        self.fc2 = nn.Linear(512 * block.expansion * args.dim, outdim)
        self.relu = nn.ReLU()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        
        return output
  
class Sparse_ResNet152(nn.Module):

    def __init__(self, outdim, save_path, Tend, eng, device, args):
        super().__init__()
        self.block = BottleNeck
        self.num_block = [3, 8, 36, 3]
        self.in_channels = 64
        self.cnn_layers = self.make_layers()
        
        self.sl1 = sparse_layer(512 * self.block.expansion, args.dim * 512 * self.block.expansion, save_path, Tend, 1, eng, device, args)
        self.sparse_layers = [self.sl1]
        self.update_mode = args.update_mode
        self.fc = nn.Linear(args.dim * 512 * self.block.expansion, outdim)
        self.clear_buffer = args.clear_buffer

    def make_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(self.block, 64, self.num_block[0], 1)
        self.conv3_x = self._make_layer(self.block, 128, self.num_block[1], 2)
        self.conv4_x = self._make_layer(self.block, 256, self.num_block[2], 2)
        self.conv5_x = self._make_layer(self.block, 512, self.num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        layers = [self.conv1, self.conv2_x, self.conv3_x, self.conv4_x, self.conv5_x, self.avg_pool]
        return nn.Sequential(*layers)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def evolve_connections(self):
        Flag = True
        for i, layer in enumerate(self.sparse_layers):
            if not self.sparse_layers[i].early_stop_signal:
                Flag = False
        
        if Flag:
            print("Early stop all the topological evolutions")
            return
            
        # remove connections
        for i, layer in enumerate(self.sparse_layers):
            layer.remove_connections()
            
        if self.update_mode == "swi":
            for i, layer in enumerate(self.sparse_layers):
                layer.num_output_active_nodes = torch.sum(torch.sum(layer.mask_after_removal, dim=0)!=0)
        
        # regrow connections
        for i, layer in enumerate(self.sparse_layers):
            layer.regrow_connections()
            if self.clear_buffer:
                layer.clear_buffers()
            layer.epoch += 1

            if layer.print_network:
                savemat(layer.adjacency_save_path + str(layer.epoch) + '.mat',
                        {"adjacency_matrix": layer.weight_mask.cpu().numpy()})

    
    def forward(self, x):
        output = self.cnn_layers(x)
        output = output.view(output.size(0), -1)
        for layer in self.sparse_layers:
            output = layer(output)
            output = self.relu(output)
        output = self.fc(output)

        return output

