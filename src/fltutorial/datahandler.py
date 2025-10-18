import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.distributions import Dirichlet
from torch.utils import data
from torchvision import datasets


@dataclass
class TransformConfig:
    """Configuration for dataset transformations."""

    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    size: Optional[int] = None
    augment: bool = False


class TransformFactory:
    """Factory for creating dataset transformations."""

    DATASET_CONFIGS = {
        "cifar10": TransformConfig(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.247, 0.243, 0.261),
            size=32,
            augment=True,
        ),
        "cifar100": TransformConfig(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), augment=True
        ),
        "mnist": TransformConfig(mean=(0.1307,), std=(0.3081,), augment=False),
        "fashionmnist": TransformConfig(mean=(0.1307,), std=(0.3081,), augment=False),
        "tinyimagenet": TransformConfig(
            mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821), augment=False
        ),
    }

    @classmethod
    def get_transforms(
        cls, dataset_name: str
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Get train and test transforms for a dataset.

        Args:
            dataset_name: Name of the dataset (case-insensitive)

        Returns:
            Tuple of (train_transform, test_transform)
        """
        config = cls.DATASET_CONFIGS.get(dataset_name.lower())
        if config is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        train_transform = cls._create_transform(config, train=True)
        test_transform = cls._create_transform(config, train=False)
        return train_transform, test_transform

    @staticmethod
    def _create_transform(
        config: TransformConfig, train: bool = True
    ) -> transforms.Compose:
        """Create transform based on dataset config."""
        transform_list = []

        # Add augmentation for training
        if train and config.augment:
            if config.size:
                transform_list.append(transforms.RandomCrop(config.size, padding=4))
            transform_list.append(transforms.RandomHorizontalFlip())

        # Base transforms
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(config.mean, config.std))

        return transforms.Compose(transform_list)


class PartitionStrategy(ABC):
    """Abstract base class for data partitioning strategies."""

    @abstractmethod
    def partition(
        self, traindata: data.Dataset, num_clients: int
    ) -> Tuple[List[data.Subset], Dict[str, List[int]]]:
        """
        Partition the training data.

        Args:
            traindata: Training dataset to partition
            num_clients: Number of clients

        Returns:
            Tuple of (list of client subsets, index mapping)
        """
        pass


class IIDPartition(PartitionStrategy):
    """Independent and Identically Distributed partitioning."""

    def partition(
        self, traindata: data.Dataset, num_clients: int
    ) -> Tuple[List[data.Subset], Dict[str, List[int]]]:
        partition_size = len(traindata) // num_clients  # type: ignore
        lengths = [partition_size] * num_clients
        client_chunks = data.random_split(
            traindata, lengths=lengths, generator=torch.Generator().manual_seed(2024)
        )

        # Store indices for each client
        idx_map = {}
        for i in range(num_clients):
            idx_map[f"client_{i}"] = client_chunks[i].indices

        logging.info(f"IID Partitioning: {partition_size} samples per client")
        return list(client_chunks), idx_map  # type: ignore


class LabelSkewPartition(PartitionStrategy):
    """Non-IID partitioning with label distribution skew using Dirichlet.

    Each client receives a non-uniform distribution of labels based on Dirichlet
    distribution. Lower alpha values create more extreme label skew.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def partition(
        self, traindata: data.Dataset, num_clients: int
    ) -> Tuple[List[data.Subset], Dict[str, List[int]]]:
        nb_samples = len(traindata)  # type: ignore
        nb_classes = len(np.unique(traindata.targets))  # type: ignore
        targets = np.array(traindata.targets)  # type: ignore
        partition_size = nb_samples // num_clients
        min_require_sample = int(partition_size * 0.1)

        logging.info(f"Label Skew - Classes: {nb_classes}, Alpha: {self.alpha}")

        min_size = 0
        it_count = 0
        idx_batch: List[List[int]] = [[] for _ in range(num_clients)]

        while min_size < min_require_sample:
            # Dirichlet distribution
            idx_batch = [[] for _ in range(num_clients)]
            lb_distribution = Dirichlet(torch.ones(num_clients) * self.alpha).sample(
                sample_shape=(nb_classes,)
            )

            for k in range(nb_classes):
                class_idx = np.where(targets == k)[0]

                # Check which clients still need samples
                check_clientsamples: List[bool] = [
                    len(client_idx) < partition_size for client_idx in idx_batch
                ]
                check_clientsamples = np.array(check_clientsamples)  # type: ignore

                # Zero out clients that already have enough samples
                lb_class_balance = lb_distribution[k] * check_clientsamples
                lb_class_balance = lb_class_balance / lb_class_balance.sum()  # type: ignore

                # Calculate cumulative proportions for splitting
                class_proportions = (
                    torch.cumsum(lb_class_balance, dim=0) * len(class_idx)  # type: ignore
                ).int()
                client_splits = np.split(class_idx, class_proportions.tolist())

                # Distribute samples to clients (excluding last split which is empty)
                for i, split in enumerate(client_splits[:-1]):
                    idx_batch[i].extend(split.tolist())

            min_size = min([len(client_idx) for client_idx in idx_batch])
            it_count += 1
            if it_count % 1000 == 0:
                logging.info(
                    f"Iteration {it_count}: Min size {min_size}, Required {min_require_sample}"
                )

        # Shuffle indices within each client
        idx_map = {}
        for i in range(num_clients):
            random.shuffle(idx_batch[i])
            idx_map[f"client_{i}"] = idx_batch[i]

        client_chunks = [
            data.Subset(traindata, idx_batch[cid]) for cid in range(num_clients)
        ]

        logging.info(
            f"Label Skew Complete: Min {min_size} samples per client, "
            f"Min Required: {min_require_sample}"
        )
        return client_chunks, idx_map


class SampleSkewPartition(PartitionStrategy):
    """Non-IID partitioning with sample quantity skew using Dirichlet."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def partition(
        self, traindata: data.Dataset, num_clients: int
    ) -> Tuple[List[data.Subset], Dict[str, List[int]]]:
        nb_samples = len(traindata)  # type: ignore
        partition_size = nb_samples // num_clients
        min_require_sample = int(partition_size * 0.05)

        logging.info(
            f"Sample Skew - Alpha: {self.alpha}, Min required: {min_require_sample}"
        )

        min_size = 0
        it_count = 0
        lb_distribution: torch.Tensor = torch.zeros(num_clients)

        while min_size < min_require_sample:
            lb_distribution = Dirichlet(torch.ones(num_clients) * self.alpha).sample()
            lb_distribution = lb_distribution / lb_distribution.sum()
            min_size = min((lb_distribution * nb_samples).int())
            it_count += 1
            if it_count % 1000 == 0:
                logging.info(
                    f"Iteration {it_count}: Min size {min_size}, Required {min_require_sample}"
                )

        # Split data according to distribution
        proportions = (torch.cumsum(lb_distribution, dim=0) * nb_samples).int()
        splits = np.split(np.arange(nb_samples), proportions)

        idx_map = {}
        for i, split in enumerate(splits[:-1]):
            idx_map[f"client_{i}"] = split.tolist()

        client_chunks = [data.Subset(traindata, splits[i]) for i in range(num_clients)]
        logging.info(f"Sample Skew Complete: Min {min_size} samples per client")
        return client_chunks, idx_map


class DataSetHandler:
    """
    Handles dataset loading, partitioning, and visualization for federated learning.

    Supports multiple partitioning strategies:
    - IID: Independent and Identically Distributed (uniform)
    - Label Skew: Non-IID with label distribution skew using Dirichlet
    - Sample Skew: Non-IID with quantity skew using Dirichlet
    """

    PARTITION_STRATEGIES = {
        "iid": IIDPartition,
        "label_skew": LabelSkewPartition,
        "sample_skew": SampleSkewPartition,
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataSetHandler.

        Args:
            config: Configuration dictionary containing:
                - data_name: Dataset name (CIFAR10, MNIST, etc.)
                - download_dir: Directory to download/store data
                - num_clients: Number of federated clients
                - validation_split: Validation split ratio
                - batch_size: Batch size for data loaders
                - alpha: Dirichlet concentration parameter for non-IID
                - partition: Partition type ('iid', 'label_skew', 'sample_skew')
                - dataloaders: Whether to create data loaders
                - partition_dir: Directory to save partition metadata
        """
        self.dataname: str = config["data_name"]
        self.data_dir: str = config["download_dir"]
        self.num_clients: int = config["num_clients"]
        self.validation_split: float = config["validation_split"]
        self.batch_size: int = config["batch_size"]
        self.alpha: float = config["alpha"]
        self.partition_type: str = config["partition"]
        self.dataloaders: bool = config["dataloaders"]
        self.partition_dir: str = config["partition_dir"]

        self.idx_map: Dict[str, List[int]] = {}
        self._partition_strategy = self._create_partition_strategy()

    def _create_partition_strategy(self) -> PartitionStrategy:
        """Create the appropriate partition strategy based on configuration."""
        strategy_class = self.PARTITION_STRATEGIES.get(self.partition_type)
        if strategy_class is None:
            raise ValueError(f"Unknown partition type: {self.partition_type}")

        # IID doesn't need alpha parameter
        if self.partition_type == "iid":
            return strategy_class()
        else:
            return strategy_class(self.alpha)

    def __call__(self) -> Tuple[List[data.Subset], data.Dataset]:
        return self.get_data()

    def __str__(self) -> str:
        return (
            f"Dataset: {self.dataname} | Data_Dir: {self.data_dir} | "
            f"Num Clients: {self.num_clients} | Validation Split: {self.validation_split}"
        )

    def plot_label_distribution(self, traindata: data.Dataset) -> None:
        """
        Generate and save visualization of label distribution across clients.

        Args:
            traindata: Training dataset with targets attribute
        """
        targets = np.array(traindata.targets)  # type: ignore
        nb_classes = len(np.unique(traindata.targets))  # type: ignore

        # Calculate label counts for each client
        client_counts = {}
        for key in self.idx_map.keys():
            indices = self.idx_map[key]
            labels = targets[indices]
            counts = np.bincount(labels, minlength=nb_classes)
            client_counts[key] = counts

        # Setup plot grid
        num_clients = len(client_counts)
        num_cols = 2
        num_rows = (num_clients + num_cols - 1) // num_cols  # Ceiling division

        # Create visualization
        plt.figure(figsize=(10, num_rows * 5))
        for i, (client, counts) in enumerate(client_counts.items()):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.bar(np.arange(nb_classes), counts)
            plt.title(f"{client} Label Distribution")
            plt.xlabel("Class")
            plt.ylabel("Count")
        plt.tight_layout()

        # Save to file
        os.makedirs(self.partition_dir, exist_ok=True)
        output_path = os.path.join(self.partition_dir, "distribution_per_client.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Label distribution plot saved to {output_path}")

    def get_data(self) -> Tuple[List[data.Subset], data.Dataset]:
        """
        Load and partition dataset for federated learning.

        Returns:
            Tuple of (list of client training datasets, test dataset)
        """
        # Get appropriate transforms for the dataset
        train_transform, test_transform = TransformFactory.get_transforms(self.dataname)

        # Load training and test datasets
        dataset_class = getattr(datasets, self.dataname)
        trainset = dataset_class(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        testset = dataset_class(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )

        num_classes = len(np.unique(trainset.targets))  # type: ignore
        logging.info(f"Dataset loaded: {self.dataname} with {num_classes} classes")

        # Partition data across clients using strategy pattern
        client_datasets, self.idx_map = self._partition_strategy.partition(
            trainset, self.num_clients
        )
        logging.info(f"Data partitioned using {self.partition_type.upper()} strategy")

        # Generate and save visualization
        self.plot_label_distribution(trainset)

        # Save partition indices
        self._save_partition_metadata()

        return client_datasets, testset

    def _save_partition_metadata(self) -> None:
        """Save partition indices to file for reproducibility."""
        os.makedirs(self.partition_dir, exist_ok=True)
        idx_map_path = os.path.join(self.partition_dir, "idx_map.json")
        with open(idx_map_path, "w") as f:
            json.dump(self.idx_map, f, indent=2)
        logging.info(f"Partition indices saved to {idx_map_path}")
