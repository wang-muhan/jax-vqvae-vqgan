import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

import jax

def get_dataloader(args, path):
    
    batch_size = args.batch_size
    assert batch_size % world_size == 0, "Batch size must be divisible by the number of processes"
    host_batch_size = batch_size // world_size
    assert host_batch_size % jax.local_device_count() == 0, "Host batch size must be divisible by the number of local devices"

    rank = jax.process_index()
    world_size = jax.process_count()

    train_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = ImageFolder(
        root = path,
        transform = train_transform
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=host_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    return train_dataloader