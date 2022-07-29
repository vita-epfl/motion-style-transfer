import os 
import yaml 
import numpy as np 
from models.trainer import YNetTrainer


def get_experiment_name(args, n_data):
    experiment = ""
    experiment += f"Seed_{args.seed}"
    if args.load_data == 'sequential':
        files = '_'.join([file.replace('.pkl', '') for file in args.train_files])
        experiment += f"__{(args.dataset_path).replace('/', '_')}_{files}"
    else:
        experiment += f"__{(args.dataset_path).replace('/', '_')}"
    experiment += f"__{args.train_net}"
    
    if args.position != []: experiment += f'__Pos_{"_".join(map(str, args.position))}' 
    if args.n_train_batch is not None: 
        experiment += f'__TrN_{n_data}'
        experiment += f'__lr_{np.format_float_positional(args.lr, trim="-")}'
        if args.smooth_val: experiment += '__smooth'
        if args.n_early_stop < args.n_epoch: experiment += f'__early_{args.n_early_stop}'
        if args.augment: experiment += '__AUG'
        if args.ynet_bias: experiment += '__bias'

    if args.network == 'original' or args.network == 'embed':
        experiment += f'__{args.network}'
    else:
        experiment += f'__fusion_{args.n_fusion}'

    return experiment


def get_params(args):
    if args.network == 'fusion': assert args.n_fusion is not None
    with open(os.path.join('config', args.config_filename)) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # load segmentation model given dataset name 
    dataset_name = params['dataset_name'].lower()

    if 'sdd' in dataset_name:
        segmentation_model = 'sdd_segmentation.pth'
    elif 'ind' in dataset_name:
        segmentation_model = 'inD_segmentation.pth'
    else:
        raise ValueError(f'Invalid {dataset_name}')
    params['segmentation_model_fp'] = os.path.join(
        params['data_dir'], params['dataset_name'], segmentation_model)

    # make n_train_batch integer 
    if hasattr(args, 'n_train_batch'):
        if args.n_train_batch is not None:
            if int(args.n_train_batch) == args.n_train_batch:
                args.n_train_batch = int(args.n_train_batch)
    
    params.update(vars(args))
    print(params)
    return params 


def get_image_and_data_path(params):
    # image path given dataset name 
    dataset_name = params['dataset_name'].lower()
    if 'sdd' in dataset_name:
        image_path = os.path.join(params['data_dir'], params['dataset_name'], 'raw', 'annotations')
    elif 'ind' in dataset_name:
        image_path = os.path.join(params['data_dir'], params['dataset_name'], 'images')
    else:
        raise ValueError(f'Invalid {dataset_name}')
    assert os.path.isdir(image_path), f'image dir error: {image_path}'
    # data path 
    data_path = os.path.join(params['data_dir'], params['dataset_name'], params['dataset_path'])
    assert os.path.isdir(data_path), f'data dir error: {data_path}'
    return image_path, data_path 


def get_position(ckpt_path, return_list=True):
    if ckpt_path is not None:
        if 'Pos' in ckpt_path:
            pos = ckpt_path.split('Pos_')[-1].split('__')[0]
            if not return_list:
                return pos
            else:
                pos_list = [i for i in pos.split('_')]
                return pos_list
        else:
            return None
    else:
        return None 


def get_ckpt_name(ckpt_path):
    ckpt_path = ckpt_path.split('/')[-1]
    train_net = ckpt_path.split('__')[2]
    if 'Pos' in ckpt_path:
        position = get_position(ckpt_path, return_list=False)
        n_train = int(ckpt_path.split('TrN_')[-1].split('_')[0])
        ckpt_name = f'{train_net}[{position}]({n_train})'
    else:
        n_train = int(ckpt_path.split('TrN_')[-1].split('_')[0])
        ckpt_name = f'{train_net}({n_train})'
    return ckpt_name 


def update_params(ckpt_path, params):
    ckpt_path = ckpt_path.split('/')[-1]
    updated_params = params.copy()
    # train net
    train_net = ckpt_path.split('__')[2].split('.')[0]
    updated_params.update({'train_net': train_net})
    # base 
    base_arch = params['pretrained_ckpt'].split('_')[-1].split('.')[0]
    if base_arch == 'embed':
        updated_params.update({'add_embedding': True})
    elif 'fusion' in base_arch:
        update_params.update({'n_fusion': int(base_arch.split('_')[-1])}) 
    # position     
    if 'Pos' in ckpt_path:
        position = get_position(ckpt_path)
        updated_params.update({'position': position})
    return updated_params


def get_ckpts_and_names(ckpts, ckpts_name, pretrained_ckpt, tuned_ckpts):
    if ckpts is not None:
        ckpts, ckpts_name = ckpts, ckpts_name
        is_file_separated = [False] * len(ckpts)
    elif pretrained_ckpt is not None:
        ckpts = [pretrained_ckpt] + tuned_ckpts
        ckpts_name = ['OODG'] + [get_ckpt_name(ckpt) for ckpt in tuned_ckpts]
        is_file_separated = [False] + [True] * len(tuned_ckpts)
    else:
        raise ValueError('No checkpoint provided')
    return ckpts, ckpts_name, is_file_separated


def restore_model(
    params, is_file_separated, base_ckpt, separated_ckpt=None):
    if not is_file_separated:
        model = YNetTrainer(params=params)
        model.load_params(base_ckpt)
    else:  
        updated_params = update_params(separated_ckpt, params)
        model = YNetTrainer(params=updated_params)
        model.load_separated_params(base_ckpt, separated_ckpt)    
    return model 
    