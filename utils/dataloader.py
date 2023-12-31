import math
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler, BatchSampler


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_data_sampler(dataset, shuffle, distributed, balanced=False, boundaries=None):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    assert not (distributed and balanced), "Cannot do weighted sampling with distributed"

    if shuffle:
        if not balanced:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            hist = np.zeros(len(boundaries))
            elem2bin = np.zeros(len(dataset.samples['trusted']), dtype=np.int)
            for si, s in enumerate(dataset.samples['trusted']):
                for level in range(len(boundaries)):
                    if boundaries[level][0] <= len(s.caption) <= boundaries[level][1]:
                        hist[level] += 1
                        elem2bin[si] = level
            valid_sizes = (hist > 0).sum()
            p_elem = np.zeros(len(dataset.samples['trusted']))
            for i, _ in enumerate(dataset.samples['trusted']):
                ind = dataset.length_sorted_trusted[i]  # due to the get item function
                #s = dataset.samples['trusted'][ind]
                bin = elem2bin[ind]
                p_elem[i] = (1 / valid_sizes) * (1 / hist[bin])

            sampler = torch.utils.data.sampler.WeightedRandomSampler(p_elem, p_elem.size)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(
        sampler, images_per_batch, num_iters=None, start_iter=0
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(
        dataset,
        batch_size,
        num_workers,
        collate_fn=None,
        max_iter=None,
        split='trainrestval',
        is_distributed=False,
        start_iter=0,
        balanced=False,
        boundaries=None,
):
    if split == 'test':
        shuffle = False
        num_iters = None
        start_iter = 0
    elif split == 'trainrestval' and max_iter is None:
        shuffle = False
        num_iters = max_iter
    else:
        shuffle = True
        num_iters = max_iter

    sampler = make_data_sampler(dataset, shuffle, is_distributed, balanced, boundaries)
    batch_sampler = make_batch_data_sampler(
        sampler, batch_size, num_iters, start_iter)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers,
        batch_sampler=batch_sampler, collate_fn=collate_fn,
    )

    return data_loader


def update_sampler_weights(data_loader, dataset, distributed, balanced=False, boundaries=None):

    assert not (distributed and balanced), "Cannot do weighted sampling with distributed"

    if balanced:
        hist = np.zeros(len(boundaries))
        elem2bin = np.zeros(len(dataset.samples['chosen']), dtype=np.int)
        for si, s in enumerate(dataset.samples['chosen']):
            for level in range(len(boundaries)):
                if boundaries[level][0] <= len(s.caption) <= boundaries[level][1]:
                    hist[level] += 1
                    elem2bin[si] = level
        valid_sizes = (hist > 0).sum()
        p_elem = np.zeros(len(dataset.samples['chosen']))
        for i, _ in enumerate(dataset.samples['chosen']):
            ind = dataset.length_sorted[i]  # due to the get item function
            bin = elem2bin[ind]
            p_elem[i] = (1 / valid_sizes) * (1 / hist[bin])

        data_loader.batch_sampler.batch_sampler.sampler.num_samples = torch.tensor(p_elem.shape[0]).cuda()
        data_loader.batch_sampler.batch_sampler.sampler.weights = torch.from_numpy(p_elem).cuda()
