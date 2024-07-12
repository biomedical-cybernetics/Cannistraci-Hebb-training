import argparse
import torch
import numpy as np
import random
from model import sparse_mlp, dense_mlp, Sparse_GoogleNet, Dense_GoogleNet, Dense_ResNet152, Sparse_ResNet152
import os 
from load_data import load_data_mlp, load_data_cnn
import torch.optim as optim
from scipy.io import loadmat, savemat
import time
from Train_and_Test import Train, Test
import random
import wandb
from sparse_topology_initialization import create_sparse_topological_initialization
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class metrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = []
        self.top1 = []
        self.top5 = []
        self.layer1_overlap_rate = []
        self.layer2_overlap_rate = []
        self.layer3_overlap_rate = []

    def update(self, loss, top1, top5):
        self.loss.append(loss)
        self.top1.append(top1)
        self.top5.append(top5)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def args():
    parser = argparse.ArgumentParser()
    # normal training arguments
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--network_structure", type=str, default="mlp")
    parser.add_argument("--cnn_structure", type=str, default = "D")
    parser.add_argument("--cuda_device", type=int, default="0")
    parser.add_argument("--check_exist", action="store_true", help="checking the experiments whether has been done or not")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--optimizer", type=str, default="sgd", help="adam, sgd...")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight_decay")
    
    
    # sparse training arguments
    parser.add_argument("--epsilon", type=float, default=0.0, help="give the sparsity to each layer by ER")
    parser.add_argument("--sparsity", type=float, default=0.99, help="directly give the sparsity to each layer")
    parser.add_argument("--update_interval", type=int, default=1, help="the number of intervals for topology evolution")
    parser.add_argument("--zeta", type=float, default=0.3, help="the fraction of removal and regrown links")
    parser.add_argument("--adaptive_zeta", action="store_true", help="add this augment to make the zeta reducing across the epochs")
    parser.add_argument("--remove_method", type=str, default="weight_magnitude", help="how to remove links, Magnitude or MEST")
    parser.add_argument("--regrow_method", type=str, default="random", help="how to regrow new links. "
                                                                            "Including: random, gradient, CH3_L3_nosubrank")
    parser.add_argument("--init_mode", type=str, default="kaiming", help="how to initialize the weights of the model."
                                                                         "Including: kaiming, xavier, gaussian, swi")
    parser.add_argument("--update_mode", type=str, default="zero", help="how to initialize the weights of the new grown links."
                                                                           "Including: kaiming, xavier, gaussian, zero, original, swi")
    parser.add_argument("--chain_removal", action="store_true", help="use forward removal and backward removal")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--print_network", action="store_true", help="save the adjacency matrix of each sandwich layer after each evolutionary epoch")
    parser.add_argument("--self_correlated_sparse", action="store_true")
    parser.add_argument("--calib_samples", type=int, default=256)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=0)
    parser.add_argument("--early_stop_threshold", type=float, default="0.9")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--WS", action="store_true")
    parser.add_argument("--BA", action="store_true")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--linearlr", action="store_true")
    parser.add_argument('--milestone', type=int, nargs='+', default=[60, 120, 160],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--discretelr", action="store_true")
    parser.add_argument("--reset_parameters", action="store_true")
    parser.add_argument("--end_factor", type=float, default=0.01)
    parser.add_argument("--T_decay", type=str, default="no_decay", choices=["no_decay", "linear"])
    parser.add_argument("--decay_factor", type=float, default=0.9)
    parser.add_argument("--clear_buffer", action="store_true")
    parser.add_argument("--old_version", action="store_true", help="use the old version of the sparse model")

    
    return parser.parse_args()


def train_model(seed, device, args):
    setup_seed(seed)

    
    print(args)
    
    # For computing the adaptive_zeta, the cosine strategy of RigL
    Tend = args.epochs//args.update_interval
    
    save_path_parts = [
        args.network_structure,
        args.dataset,
        f"s_{seed}_lr_{args.learning_rate}_e_{args.epochs}",
        f"s_{args.sparsity}_i_{args.update_interval}_az_{args.adaptive_zeta}_z_{args.zeta}_im_{args.init_mode}_um_{args.update_mode}_df_{args.decay_factor}",
    ]
    
    # Adding conditional parts
    if args.old_version:
        save_path_parts.append("old_")

    if args.self_correlated_sparse:
        save_path_parts.append("scs_")

    if args.BA:
        save_path_parts.append("ba_")

    if args.WS:
        save_path_parts.append(f"ws_beta_{args.beta}")

    if args.chain_removal:
        save_path_parts.append("chain_")

    if args.early_stop:
        save_path_parts.append("es_")
    
    if args.remove_method.split("_")[-1] == "soft":
        save_path_parts.append(f"{args.T_decay}_")
    
        
    # Adding fixed parts
    save_path_parts.append(f"d_{args.dim}_")

    save_path_parts.append(f"{args.regrow_method}_{args.remove_method}")

    # Joining all parts together to form the save path
    save_path = "/".join(save_path_parts) + "/"

    print("Save path is:", save_path)
    
    os.makedirs(save_path, exist_ok=True)

    if args.check_exist:
        if os.path.exists(save_path + "res.mat"):
            print("This simulation has already finished!!!!")
            return

    run_name = save_path
    run = wandb.init(
        # Set the project where this run will be logged
        project="{0}_{1}".format(args.dataset, args.network_structure),
        name=run_name + args.regrow_method,
        # Track hyperparameters and run metadata
        config=vars(args),
        mode="disabled" if args.no_log else "online",
    )

    
    if args.network_structure == "mlp":
        train_loader, test_loader, indim, outdim, hiddim = load_data_mlp(args)
        if args.regrow_method == "fc":
            model = dense_mlp(indim, hiddim, outdim, args).to(device)
        else:
            model = sparse_mlp(indim, hiddim, outdim, save_path, Tend, device, args).to(device)
            filename = "self-correlated_sparse/{0}".format(args.dataset)
        
            create_sparse_topological_initialization(args, model, filename=filename)
    
    elif args.network_structure == "googlenet":
        train_loader, test_loader, outdim = load_data_cnn(args)
        indim = 1024
        if args.dimension:
            hiddim = args.dimension
        else:
            hiddim = indim * args.dim
        
        if args.regrow_method == "fc":
            model = Dense_GoogleNet(indim, hiddim, outdim).to(device)
        else:
            model = Sparse_GoogleNet(indim, hiddim, outdim, save_path, Tend, device, args).to(device)
            create_sparse_topological_initialization(args, model)
            
    elif args.network_structure == "resnet152":
        train_loader, test_loader, outdim = load_data_cnn(args)
        
        if args.regrow_method == "fc":
            model = Dense_ResNet152(args, outdim).to(device)
        else:
            model = Sparse_ResNet152(outdim, save_path, Tend, device, args).to(device)
            create_sparse_topological_initialization(args, model)
    
    if args.reset_parameters:
        for layer in model.sparse_layers:
            layer.reset_parameters()
    # print(model)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay=args.weight_decay)
    if args.linearlr:
        train_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=args.end_factor, total_iters=int(args.epochs * args.decay_factor))
    elif args.discretelr:
         train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.2) #learning rate decay
        
    if args.warmup:
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)
    else:
        warmup_scheduler = None
    
    
    if args.network_structure in ["mlp", "googlenet", "resnet152"] and args.regrow_method != "fc":
        for layer in model.sparse_layers:
            layer.optimizer = optimizer

        
    m = metrics()

    for epoch in range(args.epochs):
        if epoch == 0 or args.regrow_method in ["fixed", "fc"]: 
            pass
        else:
            if epoch % args.update_interval == 0:
                print(args)
                t1 = time.time()
                model.evolve_connections()
                t2 = time.time()
                print(f'evolution time of {args.regrow_method} is: {round(t2 - t1, 2)}s')

        Train(args, model, device, train_loader, optimizer, epoch, warmup_scheduler)
        top1, top5, test_loss= Test(model, device, test_loader)
        m.update(test_loss, top1, top5)
        wandb.log({"test_accuracy": top1, "test_loss": test_loss})
        

        # print(optimizer.param_groups[0]['lr'])
        if args.discretelr:
            train_scheduler.step(epoch)
        elif args.linearlr:
            train_scheduler.step()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print("Current learning rate is:", current_lr)
        
    # save model
    savemat(save_path + "res.mat", {'top1':m.top1, 'top5':m.top5, 'loss':m.loss})




if __name__ == '__main__':
    args = args()
    
    torch.cuda.set_device(args.cuda_device)
    device = "cuda:" + str(args.cuda_device)
    
    print("using GPU: ", torch.cuda.get_device_name())
    train_model(args.seed, device, args)
