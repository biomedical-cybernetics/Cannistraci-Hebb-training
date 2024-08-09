from torchvision import datasets, transforms
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, df, target_column):
        self.labels = torch.tensor(df[target_column].values, dtype=torch.int64)
        self.features = torch.tensor(df.drop(target_column, axis=1).values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data_mlp(args):
    if args.dataset == 'EMNIST':

        # EMNIST train dataset
        train_loader = torch.utils.data.DataLoader(datasets.EMNIST(
            root='./data',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1736,), (0.3317,))
            ]),
            download=True,
            split='balanced'),
            batch_size=args.batch_size,
            shuffle=True)

        # EMNIST test dataset
        test_loader = torch.utils.data.DataLoader(datasets.EMNIST(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1736,), (0.3317,))
            ]),
            download=True,
            split='balanced'),
            batch_size=args.batch_size,
            shuffle=True)

        indim = 784
        outdim = 47
        

    elif args.dataset == "MNIST":
        # MNIST train dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True)
        # MNIST test dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True)
        indim = 784
        outdim = 10


    elif args.dataset == "Fashion_MNIST":
        # Fashion MNIST train dataset
        train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
            root='./data',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            download=True),
            batch_size=args.batch_size,
            shuffle=True)
        
        # Fashion MNIST test dataset
        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            download=True),
            batch_size=args.batch_size,
            shuffle=True)
        indim = 784
        outdim = 10
    
    elif args.dataset == "HIGGS":
        csv_file = "./data/HIGGS.csv"
        column_names = ["outcome"] + ["feature "+str(i) for i in range(1,29)]
        df = pd.read_csv(csv_file, header=None, names=column_names)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        X_df = df.iloc[:, 1:]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        df = pd.DataFrame(np.concatenate([df.iloc[:, :1].values, X_scaled], axis=1), columns=['outcome'] + list(df.columns[1:]))
        train_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(0.8*len(df))])
        train_dataset = CustomDataset(train_df, "outcome")
        test_dataset = CustomDataset(test_df, "outcome")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        indim = 28
        outdim = 2
        
    if args.dimension:
        dimension = args.dimension
    else:
        dimension = indim * args.dim
    hiddim = [dimension, dimension, dimension]
    
    return train_loader, test_loader, indim, outdim, hiddim


def load_data_cnn(args):
    if args.dataset == "ImageNet":
        workers = 4
        traindir = './data/train_raw'
        valdir = './data/test'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        )
        test_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        )


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            sampler=None
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True
        )

        outdim = 1000
            
    elif args.dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])),
            batch_size=args.batch_size, shuffle=True)
        outdim = 10
        
        
    elif args.dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762])])),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762])])),
            batch_size=args.batch_size, shuffle=True)
        outdim = 100

    return train_loader, test_loader, outdim
    