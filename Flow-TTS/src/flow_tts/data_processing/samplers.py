from abc import abstractmethod

import math
import random
import numpy as np
from torch.utils.data import Sampler
import torch
import torch.distributed as dist


class SamplerWrapper:
    def get_seeded_gen(self):
        return self.smpl.get_seeded_gen()

    def shuffle_tensor(self, t, g=None):
        return self.smpl.shuffle_tensor(t, g=g)

    def shuffle_batches(self, batches):
        self.smpl.shuffle_batches(batches)

    def filter_batches(self, batches):
        return self.smpl.filter_batches(batches)

    def __iter__(self):
        return self.smpl.__iter__()

    def __len__(self):
        return self.smpl.__len__()


class RandomSampler(SamplerWrapper):
    def __init__(self, data_source, *args, **kwargs):
        self.smpl = torch.utils.data.RandomSampler(data_source)


class SequentialSampler(SamplerWrapper):
    def __init__(self, data_source, *args, **kwargs):
        self.smpl = torch.utils.data.SequentialSampler(data_source)


class BySequenceLengthSampler(Sampler):
    def __init__(
            self,
            data_source,
            batch_size,
            max_spec_cutoff_len,
            min_spec_cutoff_len,
            batch_buckets,
            drop_last=True,
    ):
        super().__init__(data_source)
        self.data_source = data_source
        ind_n_len = []
        for i in range(len(data_source)):
            spec_len = data_source[i][1].shape[1]
            if spec_len < max_spec_cutoff_len and spec_len > min_spec_cutoff_len:
                ind_n_len.append((i, spec_len))

        self.ind_n_len = ind_n_len
        self.bucket_boundaries = np.histogram([x[1] for x in ind_n_len], bins=batch_buckets)[1]
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")

        self.boundaries = list(self.bucket_boundaries)
        self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
        self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
        self.boundaries = torch.tensor(self.boundaries)

    def __iter__(self):
        g = self.get_seeded_gen()

        data_buckets = dict()
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():

            t = self.shuffle_tensor(t=data_buckets[k], g=g)
            batch = torch.split(t, self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        self.shuffle_batches(iter_list)
        iter_list = self.filter_batches(iter_list)

        for batch in iter_list:
            for idx in batch.numpy().tolist():
                yield idx

    def element_to_bucket_id(self, seq_length):

        valid_buckets = (seq_length >= self.buckets_min) * (seq_length < self.buckets_max)
        bucket_id = valid_buckets.nonzero()[0].item()

        return bucket_id

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def filter_batches(self, batches):
        pass

    @abstractmethod
    def shuffle_batches(self, batches):
        pass

    @abstractmethod
    def shuffle_tensor(self, t, g=None):
        pass

    @abstractmethod
    def get_seeded_gen(self):
        pass


class RandomBySequenceLengthSampler(BySequenceLengthSampler):
    def get_seeded_gen(self):
        return torch.Generator()

    def shuffle_tensor(self, t, g=None):
        return t[torch.randperm(len(t))]

    def shuffle_batches(self, batches):
        random.shuffle(batches)

    def filter_batches(self, batches):
        return batches

    def __len__(self):
        return len(self.data_source)


class DistributedBySequenceLengthSampler(BySequenceLengthSampler):
    def __init__(
            self,
            data_source,
            batch_size,
            seed=0,
            rank=None,
            num_replicas=None,
            drop_last=True,
    ):
        super().__init__(data_source=data_source, batch_size=batch_size, drop_last=drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.ind_n_len) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def get_seeded_gen(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return g

    def shuffle_tensor(self, t, g=None):
        return t[torch.randperm(len(t), generator=g)]

    def shuffle_batches(self, batches):
        random.Random(self.seed + self.epoch).shuffle(batches)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def filter_batches(self, batches):
        return batches[self.rank:self.total_size:self.num_replicas]

    def __len__(self):
        return len(self.data_source) // self.num_replicas
