import random
import string
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import DEBUG
if DEBUG:
    import debug.functional as F
    
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from mixKlaus.autoaugment import CIFAR10Policy, SVHNPolicy
from mixKlaus.criterions import LabelSmoothingCrossEntropyLoss, MarginLoss
from mixKlaus.augmentation import RandomCropPaste


def get_layer_outputs(model, input):
    layer_outputs = {}

    def hook(module, input, output):
        layer_name = f"{module.__class__.__name__}_{module.parent_name}"
        layer_outputs[layer_name] = output.detach()

    # Add parent name attribute to each module
    for name, module in model.named_modules():
        module.parent_name = name

    # Register the hook to each layer in the model
    for module in model.modules():
        module.register_forward_hook(hook)

    # Pass the input through the model
    _ = model(input)

    # Remove the hooks and parent name attribute
    for module in model.modules():
        module._forward_hooks.clear()
        delattr(module, "parent_name")

    return layer_outputs


def get_criterion(args):
    if args.criterion == "ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(
                args.num_classes, smoothing=args.smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.criterion == "margin":
        criterion = MarginLoss(m_pos=0.9, m_neg=0.1, lambda_=0.5)
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion


def get_model(args):
    if args.model_name == "vit":
        from mixKlaus.vit import ViT

        net = ViT(
            args.in_c,
            args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            embed_dim=args.embed_dim,
            head=args.head,
            is_cls_token=args.is_cls_token,
        )

    elif args.model_name == "nnmf_mixer":
        from mixKlaus.vit import NNMFMixer

        net = NNMFMixer(
            conv=args.use_conv,
            dynamic_weight=args.use_dynamic_weight,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.conv_padding,
            nnmf_iterations=args.md_iter,
            nnmf_output=args.nnmf_output,
            nnmf_backward=args.nnmf_backward,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            nnmf_hidden=args.nnmf_hidden,
            nnmf_seq_len=args.nnmf_seq_len,
            embed_dim=args.embed_dim,
            gated=args.gated,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            nnmf_skip_connection=args.nnmf_skip_connection,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
            output_mode=args.output_mode,
            normalize_input=args.normalize_input,
            normalize_input_dim=args.normalize_input_dim,
            normalize_reconstruction=args.normalize_reconstruction,
            normalize_reconstruction_dim=args.normalize_reconstruction_dim,
            normalize_hidden=args.normalize_hidden,
            normalize_hidden_dim=args.normalize_hidden_dim,
            h_softmax_power=args.h_softmax_power,
            convergence_threshold=args.convergence_threshold,
        )
    elif args.model_name == "baseline_mixer":
        from mixKlaus.vit import BaselineMixer

        net = BaselineMixer(
            seq_len=args.seq_len,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            embed_dim=args.embed_dim,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
        )

    elif args.model_name == "cnn":
        from mixKlaus.cnn import BaselineCNN

        net = BaselineCNN(
            input_shape=(3, 32, 32),
            cnn_features=args.cnn_features,
            ann_layers=args.ann_layers,
        )

    elif args.model_name == "capsule_net":
        from capsule_net.network import network

        net = network(
            number_of_classes=args.num_classes,
            conv1_in_channels=args.in_c,
            conv1_out_channels=256,
            conv1_kernel_size=9,
            conv1_stride=1,
            conv2_kernel_size=9,
            conv2_stride=2,
            primary_caps_output_dim=8,
            primary_caps_output_caps=32,
            number_of_primary_caps_yx=64,
            caps_layer_output_dim=16,
            fc1_out_features=512,
            fc2_out_features=1024,
            fc3_out_features=784,
            routing_iterations=3,
        )

    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net


def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=args.size, padding=args.random_crop_padding)
    ]
    if args.dataset != "svhn":
        train_transform += [transforms.RandomHorizontalFlip()]

    if args.autoaugment:
        if args.dataset == "c10" or args.dataset == "c100":
            train_transform.append(CIFAR10Policy())
        elif args.dataset == "svhn":
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]

    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


def get_dataloader(args):
    root = "~/data"

    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes = 10
        args.size = 32
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(
            root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.CIFAR10(
            root,
            train=False,
            transform=test_transform,
            download=args.download_data,
        )

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes = 100
        args.size = 32
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(
            root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.CIFAR100(
            root,
            train=False,
            transform=test_transform,
            download=args.download_data,
        )

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes = 10
        args.size = 32
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(
            root,
            split="train",
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.SVHN(
            root,
            split="test",
            transform=test_transform,
            download=args.download_data,
        )

    elif args.dataset == "mnist":
        args.in_c = 1
        args.num_classes = 10
        args.size = 28
        args.mean, args.std = [0.1307], [0.3081]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.MNIST(
            root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.MNIST(
            root,
            train=False,
            transform=test_transform,
            download=args.download_data,
        )
    elif args.dataset == "fashionmnist":
        args.in_c = 1
        args.num_classes = 10
        args.size = 28
        args.mean, args.std = [0.2860], [0.3530]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.FashionMNIST(
            root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root,
            train=False,
            transform=test_transform,
            download=args.download_data,
        )

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return train_dl, test_dl


def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}_{args.num_layers}l"
    if not args.use_encoder_mlp:
        experiment_name += "_nem"
    if args.autoaugment:
        experiment_name += "_aa"
    if args.label_smoothing:
        experiment_name += "_ls"
    if args.rcpaste:
        experiment_name += "_rc"
    if args.cutmix:
        experiment_name += "_cm"
    if args.mixup:
        experiment_name += "_mu"

    experiment_name += f"_{random_string(5)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return experiment_name


random_string = lambda n: "".join(
    [random.choice(string.ascii_lowercase) for i in range(n)]
)


def get_experiment_tags(args):
    tags = [args.model_name]
    # add any other tags here
    return tags


class PowerSoftmax(nn.Module):
    def __init__(self, power, dim):
        super().__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        if self.power == 1:
            return F.normalize(x, p=1, dim=self.dim)
        power_x = torch.pow(x, self.power)
        return power_x / torch.sum(power_x, dim=self.dim, keepdim=True)


def anderson(f, x0, m=5, max_iter=50, tol=1e-3, lam=1e-4, beta=1.0):
    """Anderson acceleration for fixed point iteration."""

    bsz = x0.shape[0]
    X = torch.zeros(
        (bsz, m, torch.prod(torch.tensor(x0.shape[1:]))),
        dtype=x0.dtype,
        device=x0.device,
    )
    F = torch.zeros_like(X)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        try:
            alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
                :, 1 : n + 1, 0
            ]  # (bsz x n)
        except RuntimeError:
            alpha = torch.linalg.lstsq(H[:, : n + 1, : n + 1], y[:, : n + 1]).solution[
                :, 1 : n + 1, 0
            ]
        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append(
            (F[:, k % m] - X[:, k % m]).norm().item()
            / (1e-5 + F[:, k % m].norm().item())
        )
        if res[-1] < tol:
            break
    return X[:, k % m].view_as(x0), res

def get_sparsity(tensor, dim=None):
    """
    Get the sparsity of a tensor in a given dimension based on the Entropy.
    Sparsity is calculated as one minus the ratio of the actual entropy to the maximum possible entropy.
    Output is a value between zero and one as: 0 for a tensor where all values are equally likely (i.e., not sparse),
    and 1 for a tensor where one value dominates (i.e., very sparse).
    
    The entropy of a distribution is calculated as:
    H(X) = -sum(p(x) * log(p(x))) for all x in X
    
    The sparsity is then calculated as:
    S(X) = 1 - H(X) / (-log(1/N))

    Parameters:
    tensor (torch.Tensor): The input tensor.
    dim (int, optional): The dimension along which to calculate the sparsity. If None, the sparsity is calculated for the entire tensor.

    Returns:
    float: The sparsity of the tensor.
    """
    # normalize the tensor along the dimension
    probs = F.normalize(tensor.float(), p=1, dim=dim)
    entropy = torch.sum(-probs * torch.log2(probs.clamp(min=1e-20)), dim=dim)
    N = tensor.numel() if dim is None else tensor.shape[dim]
    sparsity = 1 + (entropy/torch.log2(torch.tensor(1/N))).mean()
    return sparsity