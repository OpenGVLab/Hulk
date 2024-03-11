import yaml
import logging
import numpy as np
from easydict import EasyDict as edict
import copy
import re
import torch.distributed as dist

from .utils import printlog
from torch.distributed.distributed_c10d import _get_global_rank


task_specific_param = ['backbone', 'neck', 'decoder', 'dataset', 'sampler', 'lr_scheduler', 'optimizer',
                       'extra', 'evaluation', 'model_entry_type', 'load_ignore', 'ckpt_task_id',
                       'patch_neck','patch_adapter', 'patch_proj', 'label_neck', 'label_adapter', 'label_proj',]

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def flat(nums):
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res

def specific_group_split_modality_groups(group_spec, share_backbone_group_ids,
                                   share_decoder_group_ids, share_rgb_group_ids,
                                   share_video_group_ids, share_dense_labeling_group_ids,
                                   share_sparse_labeling_group_ids, share_text_group_ids, share_modality_group_ids=None):
    ## sanity check
    assert type(group_spec) is list
    assert all(map(lambda x: type(x) is int, group_spec))

    num_groups = len(group_spec)
    splits = np.sum(group_spec)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % splits == 0, f"{world_size} % {splits}"
    unit = int(world_size / splits)

    ## split
    group_sizes = [x*unit for x in group_spec]  # [8,8,8] / [32, 16]
    groups = []
    roots = []
    last = 0
    task_info = edict()
    all_ranks = []

    for i,gs in enumerate(group_sizes):
        ranks = list(map(int, np.arange(last, last+gs)))  #[0...8], [9...15], ...
        groups.append(dist.new_group(ranks=ranks))
        roots.append(last) # 0, 8, 16
        all_ranks.append(ranks)
        if rank in ranks:  # if current gpu rank in traversed rank task group
            printlog(f">> task_info.group[{i}] ranks {ranks}")
            task_info.group = groups[-1]  # subordinate to what group
            task_info.task_size = gs  # 8
            task_info.task_id = i
            task_info.task_rank = rank - last
            task_info.task_root_rank = last
        last += gs
    task_info.root_group = dist.new_group(ranks=roots)
    printlog(f">> task_info.root_group ranks {roots}")
    task_info.task_sizes = group_sizes
    task_info.task_root_ranks = roots
    task_info.task_num = num_groups

    ## share_backbone_group spec
    if share_backbone_group_ids is not None:  # *[0,0,0]*(default) | [0,1,0]task ids
        # group size must equal within a share_group
        backboneshareid2idx = {}
        for idx, this_id in enumerate(share_backbone_group_ids):
            if this_id not in backboneshareid2idx:
                backboneshareid2idx[this_id] = list()
            backboneshareid2idx[this_id].append(idx)  # {0: [0,1,2]}| {0: [0,2], 1: [1]}

        ## create backbone share group
        for idxs in backboneshareid2idx.values():  # idxs = [0, 1, 2]
            this_group_ranks = flat([all_ranks[i] for i in idxs])
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks)
            if rank in this_group_ranks:
                task_info.backbone_share_group = this_share_group
                printlog(f">> task_info.backbone_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.backbone_group_size = len(backboneshareid2idx)
                task_info.backbone_task_size = len(backboneshareid2idx) * this_group_size
                task_info.backbone_task_rank = np.sum(rank < np.array(this_group_ranks))

    ## share_decoder_group spec
    if share_decoder_group_ids is not None:
        # group size must equal within a share_group
        decodershareid2idx = {}
        for idx, this_id in enumerate(share_decoder_group_ids):
            if this_id not in decodershareid2idx:
                decodershareid2idx[this_id] = list()
            decodershareid2idx[this_id].append(idx)

        ## create decoder share group
        for idxs in decodershareid2idx.values():
            this_group_ranks = flat([all_ranks[i] for i in idxs])
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks)
            if rank in this_group_ranks:
                task_info.decoder_share_group = this_share_group
                printlog(f">> task_info.decoder_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.decoder_group_size = len(decodershareid2idx)
                task_info.decoder_task_size = len(decodershareid2idx) * this_group_size
                task_info.decoder_task_rank = np.sum(rank < np.array(this_group_ranks))


    #  Now, only for sparse labeling to deal with the modality sharing problem,
    #  which is not a good solution, but it works.
    #  parameters that have grads in [0,1,2] are in modality share group,
    #  parameters that do not have grads in [3,4] should be set in the task-specific group.
    if share_modality_group_ids is not None:
        # group size must equal within a share_group
        modalityshareid2idx = {}
        for idx, this_id in enumerate(share_modality_group_ids):
            # -1 denotes that this modality does not appear in the current task
            # if this_id == -1:
            #     continue
            if this_id not in modalityshareid2idx:
                modalityshareid2idx[this_id] = list()
            modalityshareid2idx[this_id].append(idx)

        ## create modality share group
        for idxs in modalityshareid2idx.values():  # 0: [1,2] 1: [3]
            this_group_ranks = flat([all_ranks[i] for i in idxs]) # 1 2
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks) # 2
            if rank in this_group_ranks:
                task_info.modality_share_group = this_share_group
                printlog(f">> task_info.modality_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.modality_group_size = len(modalityshareid2idx)

    if share_rgb_group_ids is not None:
        # group size must equal within a share_group
        rgbshareid2idx = {}
        for idx, this_id in enumerate(share_rgb_group_ids):
            # -1 denotes that this modality does not appear in the current task
            # if this_id == -1:
            #     continue
            if this_id not in rgbshareid2idx:
                rgbshareid2idx[this_id] = list()
            rgbshareid2idx[this_id].append(idx)

        ## create rgb share group
        for idxs in rgbshareid2idx.values():  # 0: [1,2] 1: [3]
            this_group_ranks = flat([all_ranks[i] for i in idxs]) # 1 2
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks) # 2
            if rank in this_group_ranks:
                task_info.rgb_share_group = this_share_group
                printlog(f">> task_info.rgb_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.rgb_group_size = len(rgbshareid2idx)
                # task_info.rgb_task_size = len(rgbshareid2idx) * this_group_size
                # task_info.rgb_task_rank = np.sum(rank < np.array(this_group_ranks))
        # all_group_ranks = flat(rgbshareid2idx.values())
        # if not len(rgbshareid2idx.values()) or dist.get_rank() not in all_group_ranks:
        #     task_info.rgb_share_group = None

    if share_dense_labeling_group_ids is not None:
        # group size must equal within a share_group
        dense_labelingshareid2idx = {}
        for idx, this_id in enumerate(share_dense_labeling_group_ids):
            # -1 denotes that this modality does not appear in the current task
            # if this_id == -1:
            #     continue
            if this_id not in dense_labelingshareid2idx:
                dense_labelingshareid2idx[this_id] = list()
            dense_labelingshareid2idx[this_id].append(idx)

        ## create dense share group
        for idxs in dense_labelingshareid2idx.values():  # 0: [1,2] 1: [3]
            this_group_ranks = flat([all_ranks[i] for i in idxs]) # 1 2
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks) # 2
            if rank in this_group_ranks:
                task_info.dense_labeling_share_group = this_share_group
                printlog(f">> task_info.dense_labeling_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.dense_labeling_group_size = len(dense_labelingshareid2idx)


    if share_sparse_labeling_group_ids is not None:
        # group size must equal within a share_group
        sparse_labelingshareid2idx = {}
        for idx, this_id in enumerate(share_sparse_labeling_group_ids):
            # -1 denotes that this modality does not appear in the current task
            # if this_id == -1:
            #     continue
            if this_id not in sparse_labelingshareid2idx:
                sparse_labelingshareid2idx[this_id] = list()
            sparse_labelingshareid2idx[this_id].append(idx)

        ## create sparse share group
        for idxs in sparse_labelingshareid2idx.values():  # 0: [1,2] 1: [3]
            this_group_ranks = flat([all_ranks[i] for i in idxs]) # 1 2
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks) # 2
            if rank in this_group_ranks:
                task_info.sparse_labeling_share_group = this_share_group
                printlog(f">> task_info.sparse_labeling_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.sparse_labeling_group_size = len(sparse_labelingshareid2idx)


    if share_text_group_ids is not None:
        # group size must equal within a share_group
        textshareid2idx = {}
        for idx, this_id in enumerate(share_text_group_ids):
            # -1 denotes that this modality does not appear in the current task
            if this_id not in textshareid2idx:
                textshareid2idx[this_id] = list()
            textshareid2idx[this_id].append(idx)

        ## create text share group
        for idxs in textshareid2idx.values():  # 0: [1,2] 1: [3]
            this_group_ranks = flat([all_ranks[i] for i in idxs]) # 1 2
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks) # 2
            if rank in this_group_ranks:
                task_info.text_share_group = this_share_group
                printlog(f">> task_info.text_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.text_group_size = len(textshareid2idx)


    if share_video_group_ids is not None:
        # group size must equal within a share_group
        videoshareid2idx = {}
        for idx, this_id in enumerate(share_video_group_ids):
            # -1 denotes that this modality does not appear in the current task
            # if this_id == -1:
            #     continue
            if this_id not in videoshareid2idx:
                videoshareid2idx[this_id] = list()
            videoshareid2idx[this_id].append(idx)

        ## create video share group
        for idxs in videoshareid2idx.values():  # 0: [1,2] 1: [3]
            this_group_ranks = flat([all_ranks[i] for i in idxs]) # 1 2
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks) # 2
            if rank in this_group_ranks:
                task_info.video_share_group = this_share_group
                printlog(f">> task_info.video_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.video_group_size = len(videoshareid2idx)

    return task_info

def specific_group_split(group_spec, share_backbone_group_ids, \
                        share_neck_group_ids, share_decoder_group_ids, share_adapter_group_ids):
    ## sanity check
    assert type(group_spec) is list
    assert all(map(lambda x: type(x) is int, group_spec))

    num_groups = len(group_spec)
    splits = np.sum(group_spec)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % splits == 0, f"{world_size} % {splits}"
    unit = int(world_size / splits)

    ## split
    group_sizes = [x*unit for x in group_spec]  # [8,8,8] / [32, 16]
    groups = []
    roots = []
    last = 0
    task_info = edict()
    all_ranks = []
    # import pdb;
    # pdb.set_trace()
    for i,gs in enumerate(group_sizes):
        ranks = list(map(int, np.arange(last, last+gs)))  #[0...8], [9...15], ...
        groups.append(dist.new_group(ranks=ranks))
        roots.append(last) # 0, 8, 16
        all_ranks.append(ranks)
        if rank in ranks:  # if current gpu rank in traversed rank task group
            printlog(f">> task_info.group[{i}] ranks {ranks}")
            task_info.group = groups[-1]  # subordinate to what group
            task_info.task_size = gs  # 8
            task_info.task_id = i
            task_info.task_rank = rank - last
            task_info.task_root_rank = last
        last += gs
    task_info.root_group = dist.new_group(ranks=roots)
    printlog(f">> task_info.root_group ranks {roots}")
    task_info.task_sizes = group_sizes
    task_info.task_root_ranks = roots
    task_info.task_num = num_groups
    # pdb.set_trace()
    ## share_backbone_group spec
    if share_backbone_group_ids is not None:  # *[0,0,0]*(default) | [0,1,0]task ids
        # group size must equal within a share_group
        backboneshareid2idx = {}
        for idx, this_id in enumerate(share_backbone_group_ids):
            if this_id not in backboneshareid2idx:
                backboneshareid2idx[this_id] = list()
            backboneshareid2idx[this_id].append(idx)  # {0: [0,1,2]}| {0: [0,2], 1: [1]}

        ## create backbone share group
        for idxs in backboneshareid2idx.values():  # idxs = [0, 1, 2]
            this_group_ranks = flat([all_ranks[i] for i in idxs])
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks)
            if rank in this_group_ranks:
                task_info.backbone_share_group = this_share_group
                printlog(f">> task_info.backbone_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.backbone_group_size = len(backboneshareid2idx)
                task_info.backbone_task_size = len(backboneshareid2idx) * this_group_size
                task_info.backbone_task_rank = np.sum(rank < np.array(this_group_ranks))
    ## share_adapter_group spec
    if share_adapter_group_ids is not None:  # *[0,0,0]*(default) | [0,1,0]task ids
        # group size must equal within a share_group
        adaptershareid2idx = {}
        for idx, this_id in enumerate(share_adapter_group_ids):
            if this_id not in adaptershareid2idx:
                adaptershareid2idx[this_id] = list()
            adaptershareid2idx[this_id].append(idx)  # {0: [0,1,2]}| {0: [0,2], 1: [1]}

        ## create adapter share group
        for idxs in adaptershareid2idx.values():  # idxs = [0, 1, 2]
            this_group_ranks = flat([all_ranks[i] for i in idxs])
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks)
            if rank in this_group_ranks:
                task_info.adapter_share_group = this_share_group
                printlog(f">> task_info.adapter_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.adapter_group_size = len(adaptershareid2idx)
                task_info.adapter_task_size = len(adaptershareid2idx) * this_group_size
                task_info.adapter_task_rank = np.sum(rank < np.array(this_group_ranks))

    # pdb.set_trace()
    ## share_neck_group spec
    if share_neck_group_ids is not None:
        # group size must equal within a share_group
        neckshareid2idx = {}
        for idx, this_id in enumerate(share_neck_group_ids):
            if this_id not in neckshareid2idx:
                neckshareid2idx[this_id] = list()
            neckshareid2idx[this_id].append(idx)

        ## create neck share group
        for idxs in neckshareid2idx.values():
            this_group_ranks = flat([all_ranks[i] for i in idxs])
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks)
            if rank in this_group_ranks:
                task_info.neck_share_group = this_share_group
                printlog(f">> task_info.neck_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.neck_group_size = len(neckshareid2idx)
                task_info.neck_task_size = len(neckshareid2idx) * this_group_size
                task_info.neck_task_rank = np.sum(rank < np.array(this_group_ranks))

    ## share_decoder_group spec
    if share_decoder_group_ids is not None:
        # group size must equal within a share_group
        decodershareid2idx = {}
        for idx, this_id in enumerate(share_decoder_group_ids):
            if this_id not in decodershareid2idx:
                decodershareid2idx[this_id] = list()
            decodershareid2idx[this_id].append(idx)

        ## create decoder share group
        for idxs in decodershareid2idx.values():
            this_group_ranks = flat([all_ranks[i] for i in idxs])
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks)
            if rank in this_group_ranks:
                task_info.decoder_share_group = this_share_group
                printlog(f">> task_info.decoder_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.decoder_group_size = len(decodershareid2idx)
                task_info.decoder_task_size = len(decodershareid2idx) * this_group_size
                task_info.decoder_task_rank = np.sum(rank < np.array(this_group_ranks))
    return task_info

class Config(object):

    def __init__(self, config_file, noginfo=False, spec_ginfo_index=None):

        with open(config_file) as f:
            config = yaml.load(f, Loader=loader)
        # print('config',config)
        self.config_path = config_file

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if noginfo:
            ginfo = None
        else:  # cherrypick from tasks
            tasks = config['tasks']
            num_tasks = len(tasks)
            if spec_ginfo_index is not None:
                assert spec_ginfo_index < len(tasks), \
                'spec_ginfo_index={} is larger than num_tasks={}'.format(spec_ginfo_index, len(tasks))
                tmp_config = copy.deepcopy(config)
                config['tasks'] = dict()
                config['tasks'][0] = tmp_config['tasks'][spec_ginfo_index]
                config['tasks'][0]['gres_ratio'] = 1
                tasks = config['tasks']
                num_tasks = len(tasks)

            # parse task_common and assign to each task
            task_common = config.get('task_common', None)
            if task_common is not None:
                for i in range(num_tasks):
                    for k,v in task_common.items():
                        if not k in tasks[i]:
                            printlog('setting {} to {} for task {}'.format(k, v, i))
                            tasks[i][k] = v

            group_spec = [tasks[i].get('gres_ratio',1) for i in range(num_tasks)]

            ## share group spec
            if config['common'].get('share_backbone_group', False):
                share_backbone_group_ids = config['common']['share_backbone_group'][:num_tasks]
            else:
                share_backbone_group_ids = [0 for i in range(num_tasks)]  # hardcoded prior
            if config['common'].get('share_adapter_group', False):
                if len(config['common']['share_adapter_group']) == 1:
                    adapter_list = []
                    share_adapter_group_ids = config['common']['share_adapter_group'][:num_tasks]
            else:
                share_adapter_group_ids = [0 for i in range(num_tasks)]  # hardcoded prior

            if config['common'].get('share_neck_group', False):
                share_neck_group_ids = config['common']['share_neck_group'][:num_tasks]
            else:
                share_neck_group_ids = [0 for i in range(num_tasks)]  # hardcoded prior

            if config['common'].get('share_decoder_group', False):
                share_decoder_group_ids = config['common']['share_decoder_group'][:num_tasks]
            else:
                share_decoder_group_ids = [i for i in range(num_tasks)]  # hardcoded prior
            ginfo = specific_group_split(group_spec, share_backbone_group_ids, share_neck_group_ids,
                                     share_decoder_group_ids, share_adapter_group_ids)
            loss_weight_sum = float(np.sum(np.array([task['loss_weight'] for task in tasks.values()])))
            ginfo.task_name = tasks[ginfo.task_id]['name']
            ginfo.task_names = [tasks[i]['name'] for i in range(ginfo.task_num)]
            ginfo.task_weight = float(tasks[ginfo.task_id]['loss_weight']) / loss_weight_sum
            ginfo.task_type = tasks[ginfo.task_id].get('type', 'normal')
            ginfo.task_types = [tasks[i].get('type', 'normal') for i in range(ginfo.task_num)]
            ginfo.task_random_seed = tasks[ginfo.task_id].get('random_seed', 0)

            for p in task_specific_param:
                if p in config['tasks'][ginfo.task_id]:
                    config['common'][p] = config['tasks'][ginfo.task_id][p]
                    printlog('{} of task{} has been overided to {}'.format(p, ginfo.task_id, config['common'][p]))

        logger = logging.getLogger('global_logger')

        self.world_size = world_size
        self.rank = rank
        self.ginfo = ginfo
        self.config = config
        self.config_file = config_file

class Config_Hulk(object):

    def __init__(self, config_file, noginfo=False, spec_ginfo_index=None):

        with open(config_file) as f:
            config = yaml.load(f, Loader=loader)
        # print('config',config)
        self.config_path = config_file

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if noginfo:
            ginfo = None
        else:  # cherrypick from tasks
            tasks = config['tasks']
            num_tasks = len(tasks)
            if spec_ginfo_index is not None:
                assert spec_ginfo_index < len(tasks), \
                'spec_ginfo_index={} is larger than num_tasks={}'.format(spec_ginfo_index, len(tasks))
                tmp_config = copy.deepcopy(config)
                config['tasks'] = dict()
                config['tasks'][0] = tmp_config['tasks'][spec_ginfo_index]
                config['tasks'][0]['gres_ratio'] = 1
                tasks = config['tasks']
                num_tasks = len(tasks)

            # parse task_common and assign to each task
            task_common = config.get('task_common', None)
            if task_common is not None:
                for i in range(num_tasks):
                    for k,v in task_common.items():
                        if not k in tasks[i]:
                            printlog('setting {} to {} for task {}'.format(k, v, i))
                            tasks[i][k] = v

            group_spec = [tasks[i].get('gres_ratio',1) for i in range(num_tasks)]

            ## share group spec
            if config['common'].get('share_backbone_group', False):
                share_backbone_group_ids = config['common']['share_backbone_group'][:num_tasks]
            else:
                share_backbone_group_ids = [0 for i in range(num_tasks)]  # hardcoded prior


            if config['common'].get('share_decoder_group', False):
                share_decoder_group_ids = config['common']['share_decoder_group'][:num_tasks]
            else:
                share_decoder_group_ids = [i for i in range(num_tasks)]  # hardcoded prior

            # use modality groups to control the communication of neck, adapter, and output proj

            if config['common'].get('share_rgb_group', False):
                share_rgb_group_ids = config['common']['share_rgb_group'][:num_tasks]
            else:
                share_rgb_group_ids = [i for i in range(num_tasks)]  # hardcoded prior

            if config['common'].get('share_dense_labeling_group', False):
                share_dense_labeling_group_ids = config['common']['share_dense_labeling_group'][:num_tasks]
            else:
                share_dense_labeling_group_ids = [i for i in range(num_tasks)]

            if config['common'].get('share_sparse_labeling_group', False):
                share_sparse_labeling_group_ids = config['common']['share_sparse_labeling_group'][:num_tasks]
            else:
                share_sparse_labeling_group_ids = [i for i in range(num_tasks)]

            if config['common'].get('share_text_group', False):
                share_text_group_ids = config['common']['share_text_group'][:num_tasks]
            else:
                share_text_group_ids = [i for i in range(num_tasks)]

            if config['common'].get('share_video_group', False):
                share_video_group_ids = config['common']['share_video_group'][:num_tasks]
            else:
                share_video_group_ids = [i for i in range(num_tasks)]

            if config['common'].get('share_modality_group', False):
                share_modality_group_ids = config['common']['share_modality_group'][:num_tasks]
            else:
                share_modality_group_ids = [i for i in range(num_tasks)]

            ginfo = specific_group_split_modality_groups(group_spec, share_backbone_group_ids,
                                             share_decoder_group_ids, share_rgb_group_ids,
                                                   share_video_group_ids, share_dense_labeling_group_ids,
                                                   share_sparse_labeling_group_ids, share_text_group_ids,
                                                         share_modality_group_ids)
            loss_weight_sum = float(np.sum(np.array([task['loss_weight'] for task in tasks.values()])))
            ginfo.task_name = tasks[ginfo.task_id]['name']
            ginfo.task_names = [tasks[i]['name'] for i in range(ginfo.task_num)]
            # ginfo.task_weight = float(tasks[ginfo.task_id]['loss_weight']) / loss_weight_sum
            ginfo.task_weight = float(tasks[ginfo.task_id]['loss_weight']) 
            ginfo.task_type = tasks[ginfo.task_id].get('type', 'normal')
            ginfo.task_types = [tasks[i].get('type', 'normal') for i in range(ginfo.task_num)]
            ginfo.task_random_seed = tasks[ginfo.task_id].get('random_seed', 0)

            for p in task_specific_param:
                if p in config['tasks'][ginfo.task_id]:
                    config['common'][p] = config['tasks'][ginfo.task_id][p]
                    printlog('{} of task{} has been overided to {}'.format(p, ginfo.task_id, config['common'][p]))

        logger = logging.getLogger('global_logger')

        self.world_size = world_size
        self.rank = rank
        self.ginfo = ginfo
        self.config = config
        self.config_file = config_file
