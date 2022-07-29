import re
import argparse
import pathlib
import pandas as pd
from utils.util import get_position


def extract_train_msg(msgs):
    msg_split = re.split('save_every_n', msgs)[1:]
    df = pd.DataFrame(columns=['seed', 'pretrained_ckpt', 'experiment', 'n_param', 'n_epoch', 'ade', 'fde'])
    for msg in msg_split: 
        seed = re.search("'seed': ([\d+]),", msg)
        pretrained_ckpt = re.search("'pretrained_ckpt': '(.*?)',", msg)
        experiment = re.search("Experiment (.*?) has started", msg)
        n_param = re.search("The number of trainable parameters: ([\d]+)", msg)
        n_epoch = re.search("Early stop at epoch ([\d]+)", msg)
        # metric = re.search("Round 0: \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)", msg)
        metric = re.search('Average performance \(by [\d]+\): \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)', msg)
        df = pd.concat([df, pd.DataFrame({
            'seed': seed.group(1) if seed is not None else None,
            'pretrained_ckpt': pretrained_ckpt.group(1).split('/')[-1] if pretrained_ckpt is not None else None,
            'experiment': experiment.group(1) if experiment is not None else None,
            'n_param': n_param.group(1) if n_param is not None else 0,
            'n_epoch': n_epoch.group(1) if n_epoch is not None else 99,
            'ade': metric.group(1) if metric is not None else None,  
            'fde': metric.group(2) if metric is not None else None, }, index=[0])], ignore_index=True)
    df.seed = df.seed.astype(int)
    df.n_param = df.n_param.astype(int)
    df.n_epoch = df.n_epoch.astype(int)
    df.ade = df.ade.astype(float)
    df.fde = df.fde.astype(float)
    df['train_net'] = df['experiment'].apply(lambda x: get_train_net(x))
    df['n_train'] = df['experiment'].apply(lambda x: get_n_train(x))
    df['position'] = df['experiment'].apply(lambda x: get_position(x, return_list=False))
    df['lr'] = df['experiment'].apply(lambda x: get_lr(x))
    df['is_ynet_bias'] = df['experiment'].apply(lambda x: get_bool_bias(x))
    df['is_augment'] = df['experiment'].apply(lambda x: get_bool_aug(x))
    # reorder columns 
    reordered_cols = ['seed', 'train_net', 'n_train', 'position', 'n_param', 'n_epoch', 'lr', 'is_ynet_bias', 'is_augment', 'ade', 'fde', 'experiment', 'pretrained_ckpt']
    df = df.reindex(columns=reordered_cols)
    return df


def extract_test_msg(test_msg):
    msg_split = re.split('save_every_n', test_msg)[1:]
    df = pd.DataFrame(columns=['seed', 'pretrained_ckpt', 'tuned_ckpt', 'ade', 'fde'])
    for msg in msg_split: 
        # metric = re.search("Round 0: \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)", msg)
        metric = re.search('Average performance \(by [\d]+\): \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)', msg)
        seed = re.search("'seed': ([\d+]),", msg)
        pretrained_ckpt = re.search("'pretrained_ckpt': '(.*?)',", msg)
        tuned_ckpt = re.search("'tuned_ckpt': '(.*?)',", msg)
        df = pd.concat([df, pd.DataFrame({
            'seed': seed.group(1) if seed is not None else None,
            'pretrained_ckpt': pretrained_ckpt.group(1).split('/')[-1] if pretrained_ckpt is not None else None,
            'tuned_ckpt': tuned_ckpt.group(1).split('/')[-1] if tuned_ckpt is not None else None,
            'ade': metric.group(1) if metric is not None else None, 
            'fde': metric.group(2) if metric is not None else None}, index=[0])], )
    df.seed = df.seed.astype(int)
    df.ade = df.ade.astype(float)
    df.fde = df.fde.astype(float)
    df['train_net'] = df['tuned_ckpt'].apply(lambda x: get_train_net(x))
    df['n_train'] = df['tuned_ckpt'].apply(lambda x: get_n_train(x))#.astype(int)
    df['position'] = df['tuned_ckpt'].apply(lambda x: get_position(x, return_list=False))
    df['lr'] = df['tuned_ckpt'].apply(lambda x: get_lr(x))
    df['is_ynet_bias'] = df['tuned_ckpt'].apply(lambda x: get_bool_bias(x))
    df['is_augment'] = df['tuned_ckpt'].apply(lambda x: get_bool_aug(x))
    # reorder columns 
    reordered_cols = ['seed', 'train_net', 'n_train', 'position', 'lr', 'is_ynet_bias', 'is_augment', 'ade', 'fde', 'tuned_ckpt', 'pretrained_ckpt']
    df = df.reindex(columns=reordered_cols)
    return df


def extract_imp_msg(imp_msg):
    msg_split = re.split('save_every_n', imp_msg)[1:]
    df = pd.DataFrame(columns=['seed', 'layer', 'ade', 'fde', 'tuned_ckpt', 'pretrained_ckpt'])
    for msg in msg_split: 
        seed = re.search("'seed': ([\d+]),", msg)
        pretrained_ckpt = re.search("'pretrained_ckpt': '(.*?)',", msg)
        tuned_ckpt = re.search("'tuned_ckpts': \['(.*?)'\],", msg)
        layers = re.findall("Replacing (.*?)\n", msg)
        # metrics = re.findall("Round 0: \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)", msg)
        metrics = re.findall('Average performance \(by [\d]+\): \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)', msg)
        # temp
        df_temp = pd.DataFrame()
        df_temp['layer'] = layers 
        df_temp['metric'] = metrics
        df_temp['ade'] = df_temp.metric.apply(lambda x: x[0])
        df_temp['fde'] = df_temp.metric.apply(lambda x: x[1])
        df_temp['seed'] = seed.group(1) if seed is not None else None 
        df_temp['tuned_ckpt'] = tuned_ckpt.group(1) if tuned_ckpt is not None else None 
        df_temp['pretrained_ckpt'] = pretrained_ckpt.group(1) if pretrained_ckpt is not None else None 
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
        df.drop(columns=['metric'], inplace=True) 
    return df 


def get_train_net(ckpt_path):
    if ckpt_path is not None:
        return ckpt_path.split('__')[2]
    else:
        None 


def get_n_train(ckpt_path):
    if ckpt_path is not None:
        n_train = int(ckpt_path.split('TrN_')[-1].split('_')[0])
        return n_train
    else:
        return None 


def get_lr(ckpt_path):
    if ckpt_path is not None:
        if 'lr' in ckpt_path: 
            return ckpt_path.split('lr_')[1].split('_')[0].split('.pt')[0]
        else:
            return 0.00005
    else:
        return None 


def get_bool_bias(ckpt_path):
    if ckpt_path is not None:
        if 'bias' in ckpt_path.split('TrN')[-1]:
            return True
        else:
            return False 
    else:
        return None 


def get_bool_aug(ckpt_path):
    if ckpt_path is not None:
        if 'AUG' in ckpt_path:
            return True 
        else:
            return False 
    else:
        return None 


def extract_file(file_path, out_dir):
    with open(file_path, 'r') as f:
        msgs = f.read()
    if 'eval' in file_path:
        df = extract_test_msg(msgs)
    elif 'train' in file_path:
        df = extract_train_msg(msgs)
    elif 'imp' in file_path:
        df = extract_imp_msg(msgs)
    else:
        raise NotImplementedError
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    if '/' in file_path:
        file_name = re.search('/([^/]+).out', file_path).group(1)
    else:
        file_name = file_path.replace('.out', '')
    out_name = f'{out_dir}/{file_name}.csv'
    print(f'Saved {out_name}')
    df.to_csv(out_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default=None, type=str)
    parser.add_argument('--out_dir', default='csv/log', type=str)
    args = parser.parse_args()
    
    extract_file(args.file_path, args.out_dir)
