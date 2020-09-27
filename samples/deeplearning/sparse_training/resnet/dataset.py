import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_data():
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    # tr_data = datasets.CIFAR100('./datasets', train=True, download=True, transform=transform)
    # te_data = datasets.CIFAR100('./datasets', train=False, download=True, transform=transform)
    tr_data = datasets.CIFAR10('./datasets', train=True, download=True, transform=transform)
    te_data = datasets.CIFAR10('./datasets', train=False, download=True, transform=transform)
    print("Dataset loaded, # train: {}, # test: {}".format(len(tr_data), len(te_data)))
    return tr_data, te_data

def data_loader():
    tr_data, te_data = load_data()

    """
    sample_idx = np.arange(1000)
    tr_data = torch.utils.data.Subset(tr_data, sample_idx)
    te_data = torch.utils.data.Subset(te_data, sample_idx)
    """

    tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=64, shuffle=True, num_workers=16)
    te_loader = torch.utils.data.DataLoader(te_data, batch_size=64, shuffle=False, num_workers=16)

    return tr_loader, te_loader

