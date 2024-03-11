import os
import torch
import collections
from easydict import EasyDict as edict
import torch.distributed as dist

from torch.nn import Module
from torch.utils.data.sampler import Sampler
import math
import numpy as np
import multiprocessing as mp
import copy
import random
from collections import defaultdict

def printlog(*args, **kwargs):
    print(f"[rank {dist.get_rank()}]", *args, **kwargs)

def sync_print(*args, **kwargs):
    rank = dist.get_rank()
    # link.barrier()
    print('sync_print: rank {}, '.format(rank) + ' '.join(args), **kwargs)

class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1,
                shuffle_strategy=0, random_seed=0, imageNumPerClass=4, ret_save_path=None):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        assert rank < world_size
        sync_print('sampler: rank={}, world_size={}, random_seed={}'.format(rank, world_size, random_seed))
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter
        self.shuffle_strategy = shuffle_strategy
        self.random_seed = random_seed
        self.imageNumPerClass = imageNumPerClass
        self.ret_save_path = ret_save_path
        self.task_name = self.dataset.task_name

        self.total_size = self.total_iter*self.batch_size

        self.call = 0

        # generate indices
        if self.ret_save_path is not None:
            self.this_ret_path = os.path.join(self.ret_save_path, '_'.join([self.task_name, str(self.world_size), str(self.rank)]) + ".pth.tar")
            if os.path.exists(self.this_ret_path):
                ret_file = torch.load(self.this_ret_path)
                # ensure this task and task size is unchanged
                if ret_file['task_name'] == self.task_name and ret_file['task_size'] == self.world_size and ret_file['task_rank'] == self.rank:
                    printlog(" load task sampler from ------> {}".format(self.this_ret_path))
                    self.indices = ret_file['ret_file']
                    self.dataset.received_indices = True
                    return
            else:
                printlog("sampler file ({}) is not existed, and will be generated now--->".format(self.this_ret_path))

        if self.shuffle_strategy in [0,1,3,4,6]:
            self.indices = self.gen_new_list()
            self.dataset.indices = self.indices
            self.dataset.received_indices = True
        elif self.shuffle_strategy == 2:
            self.indices = self.gen_s2()
        elif self.shuffle_strategy == 5:
            self.indices = self.gen_s5()
        else:
            raise Error("Invalid shuffle_strategy!") # todo: undefined 'Error'???

        if self.ret_save_path is not None and not os.path.exists(self.ret_save_path):
            self.save()

    def gen_s2(self):

        np.random.seed(self.rank) # set different random seed

        indices = []
        labels = self.dataset.labels
        printlog('using shuffle strategy 2, initializing class map...')
        class2id = collections.OrderedDict()
        for i,l in enumerate(labels):
            if l in class2id:
                class2id[l].append(i)
            else:
                class2id[l] = [i]
        keys = list(class2id.keys())
        ## random shuffle class keys
        np.random.shuffle(keys)
        num_class = len(keys)
        printlog('class map done.')

        for i in range((self.last_iter+1)*self.batch_size, self.total_size):
            class_id = np.random.randint(0, num_class)
            this_num = len(class2id[keys[class_id]])
            inner_id = np.random.randint(0, this_num)
            # yield class2id[keys[class_id]][inner_id]
            indices.append(class2id[keys[class_id]][inner_id])

        return indices

    def gen_s5(self):
        np.random.seed(self.rank) # set different random seed

        indices = []
        labels = self.dataset.labels
        printlog('using shuffle strategy 5, initializing class map...')
        class2id = collections.OrderedDict()
        for i,l in enumerate(labels):
            if l in class2id:
                class2id[l].append(i)
            else:
                class2id[l] = [i]
        keys = list(class2id.keys())
        ## random shuffle class keys
        np.random.shuffle(keys)
        num_class = len(keys)
        printlog('class map done.')
        printlog('{} class with {} samples in a batch!'.format(self.batch_size//self.imageNumPerClass, self.imageNumPerClass))

        for i in range((self.last_iter+1)*self.batch_size, self.total_size):
            if i % self.imageNumPerClass == 0:
                class_id = np.random.randint(0, num_class)
                this_num = len(class2id[keys[class_id]])
            inner_id = np.random.randint(0, this_num)
            # yield class2id[keys[class_id]][inner_id]
            indices.append(class2id[keys[class_id]][inner_id])

        return indices

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices)
            # if self.shuffle_strategy in [2,5]:
            #     return self.indices
            # else:
            #     return iter(self.indices)
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle independently
        if self.shuffle_strategy == 0:

            np.random.seed(self.rank)

            indices = np.arange(len(self.dataset))
            indices = indices[:self.total_size]
            num_repeat = (self.total_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:self.total_size]

            for beg in range(0, self.total_size, len(self.dataset)):
                end = min(beg+len(self.dataset), self.total_size)
                np.random.shuffle(indices[beg:end])

        # each process shuffle all list with same seed, and pick one piece according to rank
        elif self.shuffle_strategy == 1:

            np.random.seed(self.random_seed)

            all_size = self.total_size * self.world_size
            indices = np.arange(len(self.dataset))
            indices = indices[:all_size]
            num_repeat = (all_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            np.random.shuffle(indices)
            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]

        elif self.shuffle_strategy == 3:

            np.random.seed(0)

            all_size = self.total_size * self.world_size

            ## make class map
            labels = self.dataset.labels
            class2id = collections.OrderedDict()
            for i,l in enumerate(labels):
                if l in class2id:
                    class2id[l].append(i)
                else:
                    class2id[l] = [i]
            ## calc mean_num
            class_count = [len(x) for _,x in class2id.items()]
            mean_num = int(np.mean(class_count))
            ## fill class to mean_num
            indices = []
            for _,v in class2id.items():
                if len(v) < mean_num:
                    lack_num = mean_num - len(v)
                    indices.extend(np.random.choice(v, lack_num))
                indices.extend(v)
            indices = np.array(indices)
            indices = indices[:all_size]
            printlog('using strategy 3, mean_num: {}, origin_len: {}, balanced_len: {}'.format(mean_num, len(self.dataset), len(indices)))

            num_repeat = (all_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            np.random.shuffle(indices)
            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]

        elif self.shuffle_strategy == 6:

            np.random.seed(self.random_seed)

            labels = self.dataset.labels
            printlog('using shuffle strategy 6, initializing class map...')
            class2id = collections.defaultdict(list)
            for i,l in enumerate(labels):
                class2id[l].append(i)

            mini_indices = []
            for pid, idxs in class2id.items():
                if len(idxs) < self.imageNumPerClass:
                    idxs = idxs + list(np.random.choice(idxs, size=self.imageNumPerClass-len(idxs), replace=True))
                elif len(idxs) % self.imageNumPerClass != 0:
                    add_num = int(len(idxs) // self.imageNumPerClass + 1) * self.imageNumPerClass - len(idxs)
                    idxs = idxs + list(np.random.choice(idxs, size=add_num, replace=True))

                assert len(idxs) % self.imageNumPerClass == 0
                mini_indices.extend([idxs[i:i+self.imageNumPerClass] for i in range(0, len(idxs), self.imageNumPerClass)])

            np.random.shuffle(mini_indices)
            indices = np.array(mini_indices).reshape(-1)

            all_size = self.total_size * self.world_size
            indices = indices[:all_size]
            num_repeat = (all_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]
        elif self.shuffle_strategy == 7:

            np.random.seed(self.random_seed)

            labels = self.dataset.labels
            printlog('using shuffle strategy 7, initializing class map...')
            class2id = collections.defaultdict(list)
            for i,l in enumerate(labels):
                class2id[l].append(i)

            mini_indices = []
            for pid, idxs in class2id.items():
                if len(idxs) < self.imageNumPerClass:
                    idxs = idxs + list(np.random.choice(idxs, size=self.imageNumPerClass-len(idxs), replace=True))
                elif len(idxs) % self.imageNumPerClass != 0:
                    add_num = int(len(idxs) // self.imageNumPerClass + 1) * self.imageNumPerClass - len(idxs)
                    idxs = idxs + list(np.random.choice(idxs, size=add_num, replace=True))

                assert len(idxs) % self.imageNumPerClass == 0
                mini_indices.extend([idxs[i:i+self.imageNumPerClass] for i in range(0, len(idxs), self.imageNumPerClass)])

            np.random.shuffle(mini_indices)
            indices = np.array(mini_indices).reshape(-1)

            all_size = self.total_size * self.world_size
            indices = indices[:all_size]
            num_repeat = (all_size-1) // indices.shape[0] + 1

            for repeat in range(num_repeat)-1:
                for pid, idxs in class2id.items():
                    if len(idxs) < self.imageNumPerClass:
                        idxs = idxs + list(np.random.choice(idxs, size=self.imageNumPerClass-len(idxs), replace=True))
                    elif len(idxs) % self.imageNumPerClass != 0:
                        add_num = int(len(idxs) // self.imageNumPerClass + 1) * self.imageNumPerClass - len(idxs)
                        idxs = idxs + list(np.random.choice(idxs, size=add_num, replace=True))

                    assert len(idxs) % self.imageNumPerClass == 0
                    mini_indices.extend([idxs[i:i+self.imageNumPerClass] for i in range(0, len(idxs), self.imageNumPerClass)])

            np.random.shuffle(mini_indices)
            indices = np.array(mini_indices).reshape(-1)

            all_size = self.total_size * self.world_size
            indices = indices[:all_size]

            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]
        elif self.shuffle_strategy == 8:

            np.random.seed(self.random_seed)

            labels = self.dataset.labels
            print('using shuffle strategy 8, initializing class map...')
            class2id = collections.defaultdict(list)
            for i,l in enumerate(labels):
                class2id[l].append(i)

            pids = set()
            cls_pids_idxs = collections.defaultdict(list)
            for pid, idxs in class2id.items():
                if len(idxs) < self.imageNumPerClass:
                    idxs = idxs + list(np.random.choice(idxs, size=self.imageNumPerClass-len(idxs), replace=True))
                elif len(idxs) % self.imageNumPerClass != 0:
                    add_num = int(len(idxs) // self.imageNumPerClass + 1) * self.imageNumPerClass - len(idxs)
                    idxs = idxs + list(np.random.choice(idxs, size=add_num, replace=True))

                assert len(idxs) % self.imageNumPerClass == 0
                np.random.shuffle(idxs)
                idxs = [idxs[i:(i+self.imageNumPerClass)] for i in range(0, len(idxs), self.imageNumPerClass)]
                ptr = 0
                values = [ptr, idxs]
                cls_pids_idxs[pid] = values
                pids.add(pid)

            indices = []
            classnum_per_batch = self.batch_size // self.imageNumPerClass
            while len(pids) >= classnum_per_batch:
                batch = []
                sub_pids = []
                for i, pid in enumerate(pids):
                    if i >= classnum_per_batch:
                        break
                    sub_pids.append(pid)

                for pid in sub_pids:
                    ptr = cls_pids_idxs[pid][0]
                    idxs = cls_pids_idxs[pid][1]
                    batch.extend(idxs[ptr])
                    if ptr + 1 >= len(idxs):
                        pids.remove(pid)
                    else:
                        cls_pids_idxs[pid][0] += 1
                indices.append(batch)

            np.random.shuffle(indices)
            indices = np.array(indices).reshape(-1)

            all_size = self.total_size * self.world_size
            indices = indices[:all_size]
            num_repeat = (all_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]
        else:
            raise RuntimeError('unknow shuffle strategy')

        assert len(indices) == self.total_size

        return indices[(self.last_iter+1)*self.batch_size:]

    def __len__(self):
        return self.total_size - (self.last_iter+1)*self.batch_size

    def save(self):
        torch.save({'task_name': self.task_name,
                    'task_size': self.world_size,
                    'task_rank': self.rank,
                    'ret_file': self.indices}, self.this_ret_path)
        printlog("save sampler file  ------> {}".format(self.this_ret_path))

class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, world_size=None, rank=None):
        if world_size == None:
            world_size = dist.get_world_size()
        if rank == None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        assert len(self.dataset) >= self.world_size, f'{len(self.dataset)} vs {self.world_size}'
        sub_num = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        self.beg = sub_num * self.rank
        self.end = min(self.beg+sub_num, len(self.dataset))

    def __iter__(self):
        indices = list(range(self.beg, self.end))
        return iter(indices)

    def __len__(self):
        return self.end - self.beg