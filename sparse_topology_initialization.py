import torch
from torchvision import datasets, transforms
from scipy.io import loadmat, savemat
import numpy as np
import random

def load_calib_dataset(args, data_dir='./data'):
    if args.dataset == "MNIST":
        dataloader = torch.utils.data.DataLoader(
                        datasets.MNIST(data_dir, train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ])),
                        batch_size=args.calib_samples, shuffle=True)
        input_of_sparse_layer = np.zeros((784,60000))
    elif args.dataset == "Fashion_MNIST":
        dataloader= torch.utils.data.DataLoader(datasets.FashionMNIST(
                    root=data_dir,
                    train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ]),
                    download=True),
                    batch_size=args.calib_samples,
                    shuffle=True)
        input_of_sparse_layer = np.zeros((784,60000))
    elif args.dataset == "EMNIST":
        dataloader = torch.utils.data.DataLoader(datasets.EMNIST(
                    root=data_dir,
                    train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ]),
                    download=True,
                    split='balanced'),
                    batch_size=args.calib_samples,
                    shuffle=True)
        input_of_sparse_layer = np.zeros((784,112800))
    return dataloader, input_of_sparse_layer
        

def create_sparse_topological_initialization(args, model, filename=None):
    
    if args.self_correlated_sparse:
        dataloader, input_of_sparse_layer = load_calib_dataset(args, data_dir='./data')

        print("Using self correlated sparse of mlp!!!")
        
        import os
        if os.path.exists(filename):
            corr = loadmat(filename + "/corr.mat")["corr"]
        else:
            for batch_idx, (data, _) in enumerate(dataloader):
                input_of_sparse_layer[:,batch_idx*args.calib_samples:batch_idx*args.calib_samples + data.shape[0]] = data.reshape(-1, 784).numpy().transpose(1, 0)
            corr = np.corrcoef(input_of_sparse_layer)
            os.makedirs(filename)
            print("done")
            
            savemat(filename + "/corr.mat", {"corr":corr})
        create_self_correlated_sparse(model, corr, args.dim)
    
    elif args.BA:
        for layer in model.sparse_layers:
            create_ba_sparse(layer) 

    elif args.WS:
        for layer in model.sparse_layers:
            create_ws_sparse(layer, args)
            
    

            
def create_ws_sparse(layer, args):
    K = (1- layer.sparsity) * layer.indim * layer.outdim / (layer.indim + layer.outdim)
    
    K1 = int(K)
    K2 = int(K) + 1
    dim = max(layer.outdim, layer.indim)
    my_list = [K1] * int(dim * (K2 - K)) + [K2] * int(dim * (K-K1) + 1)
    random.shuffle(my_list)
    
    adj = np.zeros((layer.indim, layer.outdim))

    rate = layer.outdim/layer.indim
    for i in range(layer.indim):
        idx = [(int(i*rate) + j) % layer.outdim for j in range(my_list[i])]
        adj[i, idx] = 1 
    rate = layer.indim/layer.outdim
    random.shuffle(my_list)
    for i in range(layer.outdim):
        idx = [(int(i*rate) + j + 1) % layer.indim for j in range(my_list[i])]
        adj[idx, i] = 1 
        
    # rewiring
    if args.beta != 0:
        randomness = np.random.binomial(1, p=args.beta, size=int(np.sum(adj)))
        # print(randomness)
        count = 0
        for i in range(layer.indim):
            for j in range(layer.outdim):
                if adj[i][j] == 1:
                    if randomness[count] == 1:
                        adj[i][j] = 0
                    
                    count += 1
        
        # regrow
        noRewires = layer.n_params - np.sum(adj)
        nrAdd = 0
        while (nrAdd < noRewires):
            i = np.random.randint(0, layer.indim)
            j = np.random.randint(0, layer.outdim)
            if adj[i][j] == 0:
                nrAdd += 1
                adj[i][j] = 1
        
        print(np.sum(adj), noRewires)
    layer.weight_mask = torch.LongTensor(adj).to(layer.device)




def generate_barabasi_alberta_graph(N, m):
    adj = np.zeros((N, N))
    if not isinstance(m, int):
        print("m is not an integer")
        m1 = int(m)
        m2 = int(m) + 1
        
        adj[:m2, :m2] = np.triu(np.ones((m2, m2)), k=1) + np.triu(np.ones((m2, m2)), k=1).T
        my_list = [m1] * int((N-m2) * (m2 - m)) + [m2] * int((N-m2) * (m-m1) + 1)
        random.shuffle(my_list)
        for i in range(m2, N):
            targets = np.arange(i)
            p_normalized = np.sum(adj[:i, :i], axis=1) / np.sum(np.sum(adj[:i, :i], axis=1))

            
            m_tmp = my_list[i-m2]
            idx = np.random.choice(targets, size=m_tmp, replace=False, p=p_normalized)
            adj[i, idx] = 1
            adj[idx, i] = 1
            
    return adj 

def create_ba_sparse(layer):
    m = (1- layer.sparsity) * layer.indim * layer.outdim / (layer.indim + layer.outdim)
    adj = generate_barabasi_alberta_graph(layer.indim + layer.outdim, m)
    nodes = list(range(layer.indim + layer.outdim))
    random.shuffle(nodes)
    
    layer_N = list(set(nodes[:layer.indim]))
    layer_M = list(set(nodes[layer.indim:]))

    adj = adj[layer_N+layer_M]
    adj = adj[:, layer_N+layer_M]
    
    adj = np.triu(adj, k=1)
    final_adj = adj[:layer.indim, layer.indim:]
    layer1_fru = np.array(adj[:layer.indim, :layer.indim].nonzero()).reshape(-1)
    layer2_fru = np.array(adj[layer.indim:, layer.indim:].nonzero()).reshape(-1)
    np.random.shuffle(layer1_fru)
    np.random.shuffle(layer2_fru)

    if len(layer1_fru) <= len(layer2_fru):
        # print("in")
        layer2_fru_flag = np.zeros_like(layer2_fru)
        for i in range(len(layer1_fru)):
            for j in range(len(layer2_fru)):
                if layer2_fru_flag[j] == 0:
                    if final_adj[layer1_fru[i], layer2_fru[j]]:
                        continue
                    else:
                        final_adj[layer1_fru[i], layer2_fru[j]] = 1
                        layer2_fru_flag[j] = 1
                        break
                else:
                    continue
             
        zero_indices = np.where(layer2_fru_flag == 0)[0]
        layer2_r = layer2_fru[zero_indices]
        for i in range(len(layer2_r)//2):
            node_degrees = np.sum(final_adj, axis=1)
            prob_distribution = node_degrees / np.sum(node_degrees)
            while True:
                target_node = np.random.choice(np.arange(layer.indim), size=1, replace=False, p=prob_distribution)
                if final_adj[target_node, layer2_r[i]] == 1:
                    continue
                else:
                    final_adj[target_node, layer2_r[i]] = 1
                    break
                
    elif len(layer1_fru) > len(layer2_fru):
        layer1_fru_flag = np.zeros_like(layer1_fru)
        for i in range(len(layer2_fru)):
            for j in range(len(layer1_fru)):
                if layer1_fru_flag[j] == 0:
                    if final_adj[layer1_fru[j], layer2_fru[i]]:
                        continue
                    else:
                        final_adj[layer1_fru[j], layer2_fru[i]] = 1
                        layer1_fru_flag[j] = 1
                        break
                else:
                    continue
        zero_indices = np.where(layer1_fru_flag == 0)[0]
        layer1_r = layer1_fru[zero_indices]
        for i in range(len(layer1_r)//2):
            node_degrees = np.sum(final_adj, axis=0)
            prob_distribution = node_degrees / np.sum(node_degrees)
            while True:
                target_node = np.random.choice(np.arange(layer.outdim), size=1, replace=False, p=prob_distribution)
                if final_adj[layer1_r[i], target_node] == 1:
                    continue
                else:
                    final_adj[layer1_r[i], target_node] = 1
                    break
    print(int(np.sum(final_adj)))
    layer.weight_mask = torch.Tensor(final_adj).to(layer.device)
    
def create_self_correlated_sparse(model, corr, dim):
    isnan = np.isnan(corr)
    corr[isnan] = 0
    for i in range(corr.shape[0]):
        corr[i, i] = 0
    
    # 1x of the dimension
    if dim == 1:
        for i in range(len(model.sparse_layers)):
            number_of_links = model.sparse_layers[i].n_params
            update_topology(model.sparse_layers[i], corr, number_of_links)

    # 2x of the dimension
    elif dim == 2:
        dimension = corr.shape[0] * 2
        expanded_dimension = np.zeros((dimension, dimension))
        expanded_dimension[:dimension//2, :dimension//2] = corr
        expanded_dimension[:dimension//2, dimension//2:] = corr
        expanded_dimension[dimension//2:, :dimension//2] = corr
        expanded_dimension[dimension//2:, dimension//2:] = corr
        
        for i in range(len(model.sparse_layers)):
            number_of_links = model.sparse_layers[i].n_params
            if i == 0:
                first_layer = expanded_dimension[:dimension//2, :].copy()
                update_topology(model.sparse_layers[i], first_layer, number_of_links)
            else:
                update_topology(model.sparse_layers[i], expanded_dimension, number_of_links)
    
def update_topology(layer, corr, number_of_links):
    adj = torch.zeros_like(torch.Tensor(corr))
    corr_flatten = torch.abs(torch.Tensor(corr).flatten())
    
    threshold = torch.abs(torch.sort(-torch.abs(corr_flatten))[0][number_of_links-1])
    corr = torch.Tensor(corr)
    adj[torch.abs(corr)>=threshold]=1
    adj[torch.abs(corr)<threshold]=0

    layer.weight_mask = adj.to(layer.device)
    