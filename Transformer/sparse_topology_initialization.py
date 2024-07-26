import torch
from torchvision import datasets, transforms
from scipy.io import loadmat, savemat
import numpy as np
import random
import torch.nn.functional as F

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

def rewire_connections(layer):
    new_matrix = torch.zeros_like(layer.weight_mask).to(layer.device)

    # 获取矩阵的列数和行数
    cols = new_matrix.shape[1]
    rows = new_matrix.shape[0]

    # 对每一列进行处理
    for i in range(cols):
        # 当前列
        column = layer.weight_mask[:, i]
        
        # 计算当前列中非零元素的数量
        num_connections = column.nonzero().numel()
        
        # 生成新的随机位置
        new_positions = torch.randperm(rows)[:num_connections]
        
        # 在新的随机位置设置为1或原始非零值
        new_matrix[new_positions, i] = column[column != 0]

    layer.weight_mask = new_matrix
                
                
    

def generate_bipartite_npso(a: int, b: int, m: float, t: float, gamma: float, c: int, reconnect_mode: str, eng):

    x, coords, a_set, b_set, comm, d = eng.bipartite_nPSO(float(a), float(b), m, float(t), gamma, float(c),
                                                            reconnect_mode, nargout=6)
    return np.array(x), np.array(a_set, dtype=int).squeeze()-1, np.array(b_set, dtype=int).squeeze()-1


def create_sparse_topological_initialization(args, model, filename=None, eng=None):
    
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
        create_self_correlated_sparse(model, corr, args.dim, args.soft_csti, args.noise_csti)
    
    elif args.BA:
        for i, layer in enumerate(model.sparse_layers):
            create_ba_sparse(layer) 
            if i == 0 and args.rewire_first_layer:
                rewire_connections(layer)
            

    elif args.WS:
        for layer in model.sparse_layers:
            create_ws_sparse(layer, args)
            
    elif args.load_existing_topology:
        for i, layer in enumerate(model.sparse_layers):
            adj = loadmat(f"{args.load_existing_topology}/{i}.mat")["adjacency_matrix"]
            layer.weight_mask = torch.Tensor(adj).to(layer.device)
            
    elif args.nPSO:
        
        for i, layer in enumerate(model.sparse_layers):
            m = (1- layer.sparsity) * layer.indim * layer.outdim / (layer.indim + layer.outdim)
            if i == 0:
                x, a_sets, b_sets = generate_bipartite_npso(layer.indim, layer.outdim, m, args.temperature, args.gamma_nPSO,
                                                                args.communities, args.reconnect_mode, eng=eng)
            else:
                x, a_sets, b_sets = generate_bipartite_npso(layer.indim, layer.outdim, m, args.temperature, args.gamma_nPSO,
                                                                args.communities, "none", eng=eng)
            # print(x.shape, a_sets.shape, b_sets.shape)
            # print(torch.sum(torch.Tensor(x)))
            # tmp_x = x.copy()
            # tmp_x[:layer.indim, :]=x[a_sets]
            # tmp_x[layer.indim:, :]=x[b_sets]
            
            # x = tmp_x
            # tmp_x = x.copy()
            # tmp_x[:, :layer.indim]=x[:, a_sets]
            # tmp_x[:, layer.indim:]=x[:, b_sets]
            # tmp_x[]
            layer.weight_mask = torch.Tensor(x).to(layer.device)
            print(torch.sum(layer.weight_mask).item(), layer.weight_mask.shape)
        # exit()
            
    
    if args.soft_resort or args.rigid_resort:
        for i in range(len(model.sparse_layers)-1):
            out_neuron_degree = torch.sum(model.sparse_layers[i].weight_mask, dim=0, dtype=torch.float32)
            in_neuron_degree = torch.sum(model.sparse_layers[i+1].weight_mask, dim=1, dtype=torch.float32)
            
            if args.rigid_resort:
                out_neuron_idx = torch.sort(out_neuron_degree)[1]
                in_neuron_idx = torch.sort(in_neuron_degree)[1]
                model.sparse_layers[i].weight_mask = model.sparse_layers[i].weight_mask[:, out_neuron_idx]
                model.sparse_layers[i+1].weight_mask = model.sparse_layers[i+1].weight_mask[in_neuron_idx, :]
                
            elif args.soft_resort:
                out_neuron_idx = torch.sort(out_neuron_degree)[1]
                model.sparse_layers[i].weight_mask = model.sparse_layers[i].weight_mask[:, out_neuron_idx]
                in_neuron_idx = soft_resort(in_neuron_degree)
                model.sparse_layers[i+1].weight_mask = model.sparse_layers[i+1].weight_mask[in_neuron_idx, :]
                
def soft_resort(in_neuron_degree):
    sampled_indices = torch.multinomial(in_neuron_degree, num_samples=in_neuron_degree.shape[0], replacement=False)
    return sampled_indices


def create_ws_sparse(layer, args):
    indim = min(layer.indim, layer.outdim)
    outdim = max(layer.indim, layer.outdim)
    K = (1- layer.sparsity) * indim * outdim / (indim + outdim)
    
    K1 = int(K)
    K2 = int(K) + 1
    dim = max(outdim, indim)
    my_list = [K1] * int(dim * (K2 - K)) + [K2] * int(dim * (K-K1) + 1)
    random.shuffle(my_list)
    
    adj = np.zeros((indim, outdim))

    rate = outdim/indim
    for i in range(indim):
        idx = [(int(i*rate) + j) % outdim for j in range(my_list[i])]
        adj[i, idx] = 1 
    rate = indim/outdim
    random.shuffle(my_list)
    for i in range(outdim):
        idx = [(int(i*rate) + j + 1) % indim for j in range(my_list[i])]
        adj[idx, i] = 1 
        
    # rewiring
    if args.ws_beta != 0:
        randomness = np.random.binomial(1, p=args.ws_beta, size=int(np.sum(adj)))
        # print(randomness)
        count = 0
        for i in range(indim):
            for j in range(outdim):
                if adj[i][j] == 1:
                    if randomness[count] == 1:
                        adj[i][j] = 0
                    
                    count += 1
        
        # regrow
        noRewires = int(layer.sparsity * indim * outdim) - np.sum(adj)
        nrAdd = 0
        while (nrAdd < noRewires):
            i = np.random.randint(0, indim)
            j = np.random.randint(0, outdim)
            if adj[i][j] == 0:
                nrAdd += 1
                adj[i][j] = 1
        
        print(np.sum(adj), noRewires)

    if layer.indim != indim:
        layer.weight_mask = torch.Tensor(adj).to(layer.device).t()
    else:
        layer.weight_mask = torch.Tensor(adj).to(layer.device)
    



def create_ws_sparse_scheduler(sparsity, w, args):
    indim = min(w.shape[0], w.shape[1])
    outdim = max(w.shape[0], w.shape[1])
    K = (1- sparsity) * indim * outdim / (indim + outdim)
    
    K1 = int(K)
    K2 = int(K) + 1
    dim = max(outdim, indim)
    my_list = [K1] * int(dim * (K2 - K)) + [K2] * int(dim * (K-K1) + 1)
    random.shuffle(my_list)
    
    adj = np.zeros((indim, outdim))

    rate = outdim/indim
    for i in range(indim):
        idx = [(int(i*rate) + j) % outdim for j in range(my_list[i])]
        adj[i, idx] = 1 
    rate = indim/outdim
    random.shuffle(my_list)
    for i in range(outdim):
        idx = [(int(i*rate) + j + 1) % indim for j in range(my_list[i])]
        adj[idx, i] = 1 
        
    # rewiring
    if args.ws_beta != 0:
        randomness = np.random.binomial(1, p=args.ws_beta, size=int(np.sum(adj)))
        # print(randomness)
        count = 0
        for i in range(indim):
            for j in range(outdim):
                if adj[i][j] == 1:
                    if randomness[count] == 1:
                        adj[i][j] = 0
                    
                    count += 1
        
        # regrow
        noRewires = int(sparsity * indim * outdim) - np.sum(adj)
        nrAdd = 0
        while (nrAdd < noRewires):
            i = np.random.randint(0, indim)
            j = np.random.randint(0, outdim)
            if adj[i][j] == 0:
                nrAdd += 1
                adj[i][j] = 1
        
        print(np.sum(adj), noRewires)
    if w.shape[0] != indim:
        return torch.LongTensor(adj).to(w.device).t()

    return torch.LongTensor(adj).to(w.device)
    # layer.weight_mask = torch.LongTensor(adj).to(layer.device)




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
    
def create_self_correlated_sparse(model, corr, dim, soft=False, noise=False):
    isnan = np.isnan(corr)
    corr[isnan] = 0
    for i in range(corr.shape[0]):
        corr[i, i] = 0
    
    if noise:
        corr += np.random.randn(corr.shape[0], corr.shape[1])
    # 1x of the dimension
    if dim == 1:
        for i in range(len(model.sparse_layers)):
            number_of_links = model.sparse_layers[i].n_params
            update_topology(model.sparse_layers[i], corr, number_of_links, soft)

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
                update_topology(model.sparse_layers[i], first_layer, number_of_links, soft)
            else:
                update_topology(model.sparse_layers[i], expanded_dimension, number_of_links, soft)
    
def update_topology(layer, corr, number_of_links, soft=False):
    adj = torch.zeros_like(torch.Tensor(corr))
    corr_flatten = torch.abs(torch.Tensor(corr).flatten())
    if soft:
        probabilities = corr_flatten / corr_flatten.sum()
        sampled_flat_indices = torch.multinomial(probabilities, number_of_links, replacement=False)
        adj = adj.reshape(-1)
        adj[sampled_flat_indices] = 1
        adj = adj.reshape(corr.shape[0], corr.shape[1])
    else:
        threshold = torch.abs(torch.sort(-torch.abs(corr_flatten))[0][number_of_links-1])
        corr = torch.Tensor(corr)
        adj[torch.abs(corr)>=threshold]=1
        adj[torch.abs(corr)<threshold]=0

    layer.weight_mask = adj.to(layer.device)
    