import re
import argparse
import pathlib
import numpy as np
import pandas as pd
from utils.util import get_position
import matplotlib.pyplot as plt


def moving_average(x, window, mode='same', box_loc='middle'):
    if box_loc == 'middle':
        data = np.convolve(x, np.ones(window), mode) / window
        n = x.shape[0]
        adjust = window // 2
        for i in range(adjust):
            # first several points
            data[i] = np.mean(data[:(i+adjust+1)])
            # last several points 
            data[n-i-1] = np.mean(x[(n-i-adjust-1):])
        return data 
    elif box_loc == 'history':
        data = np.zeros(x.shape[0])
        for i in range(window-1):
            data[i] = np.mean(x[:i+1])
        for i in range(window-1, x.shape[0]):
            data[i] = np.mean(x[(i-window+1):(i+1)])
        return data 
    else:
        raise NotImplementedError


def extract_training_score(text):
    df = pd.DataFrame()
    for row in re.findall('Epoch ([\d]+): 	Train \(Top-1\) ADE: ([\d\.]+) FDE: ([\d\.]+) 		Val \(Top-k\) ADE: ([\d\.]+) FDE: ([\d\.]+)', text):
        d = {'epoch': row[0], 'train_ade': row[1], 'train_fde': row[2], 'val_ade': row[3], 'val_fde': row[4]}
        df = pd.concat([df, pd.DataFrame(d, index=[0])], ignore_index=True)
    df.epoch = df.epoch.astype(int)
    df.train_ade = df.train_ade.astype(float)
    df.train_fde = df.train_fde.astype(float)
    df.val_ade = df.val_ade.astype(float)
    df.val_fde = df.val_fde.astype(float)
    return df


def extract_curve_seed(
    train_msgs, test_msgs=None, val_window=9, test_window=9, 
    show_raw_val=False, show_raw_test=False, box_loc='middle',
    out_path='figures/training_curve/curve.png'):
    if test_msgs is not None:
        df_test = extract_test_score(test_msgs)

    train_msgs_list = re.split('save_every_n', train_msgs)[1:]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for msg in train_msgs_list: 
        df_curve = extract_training_score(msg)

        experiment = re.search("Experiment (.*?) has started", msg).group(1)
        train_seed = int(re.search("'seed': ([\d+]),", msg).group(1))
        n_epoch = re.search("Early stop at epoch ([\d]+)", msg)
        n_epoch = int(n_epoch.group(1)) - 30 if n_epoch is not None else df_curve.val_ade.argmin()
        metric = re.search('Average performance \(by [\d]+\): \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)', msg)
        ade = round(float(metric.group(1)), 2)
        fde = round(float(metric.group(2)), 2)
        train_net = get_train_net(experiment)
        n_train = int(get_n_train(experiment))
        position = get_position(experiment, return_list=False)
        
        if test_msgs is not None:
            df_test_seed = df_test[df_test.train_seed == train_seed]
        
        if position is not None:    
            label_name = f'TrS{train_seed}_{train_net}[{position}]({n_train})_{ade}/{fde}'
        else:
            label_name = f'TrS{train_seed}_{train_net}({n_train})_{ade}/{fde}'
        
        val_ade = df_curve.val_ade
        val_fde = df_curve.val_fde
        if test_msgs is not None:
            test_ade = df_test_seed.ade 
            test_fde = df_test_seed.fde 
            test_ready = True if test_ade.shape[0] != 0 else False 
        if val_window is not None:
            val_ade = moving_average(df_curve.val_ade, val_window, box_loc=box_loc)
            val_fde = moving_average(df_curve.val_fde, val_window, box_loc=box_loc)
            if test_msgs is not None and test_ready:
                test_ade = moving_average(test_ade, test_window, box_loc='middle')
                test_fde = moving_average(test_fde, test_window, box_loc='middle')            

        start = 5
        # ade 
        # validation
        p = axes[0].plot(df_curve.epoch[start:], val_ade[start:], linewidth=1)
        axes[0].scatter(n_epoch, val_ade[n_epoch], c=p[-1].get_color(), marker='*')
        print(f'Train_seed={train_seed}, epoch_stop={n_epoch}')
        if show_raw_val:
            axes[0].plot(df_curve.epoch[start:], df_curve.val_ade[start:], 
                linewidth=0.5, alpha=0.5, c=p[-1].get_color())
            axes[0].scatter(df_curve.epoch[start:], df_curve.val_ade[start:], c=p[-1].get_color(), s=1)
        # test 
        if test_ready:
            if show_raw_test:
                axes[0].plot(df_test_seed.epoch[start:], df_test_seed.ade[start:], 
                    linewidth=0.5, alpha=0.5, c=p[-1].get_color())
                axes[0].scatter(df_test_seed.epoch[start:], df_test_seed.ade[start:], 
                    c=p[-1].get_color(), s=1)
            axes[0].plot(df_test_seed.epoch[start:], test_ade[start:], 
                c=p[-1].get_color(), linewidth=2.5)

        # fde 
        # validation 
        p = axes[1].plot(df_curve.epoch[start:], val_fde[start:], 
            label=label_name, linewidth=1)
        axes[1].scatter(n_epoch, val_fde[n_epoch], c=p[-1].get_color(), marker='*')
        if show_raw_val:
            axes[1].plot(df_curve.epoch[start:], df_curve.val_fde[start:], 
                linewidth=0.5, alpha=0.5, c=p[-1].get_color())
            axes[1].scatter(df_curve.epoch[start:], df_curve.val_fde[start:], 
                c=p[-1].get_color(), s=1)
        # test 
        if test_ready:
            if show_raw_test:
                axes[1].plot(df_test_seed.epoch[start:], df_test_seed.fde[start:], 
                    linewidth=0.5, alpha=0.5, c=p[-1].get_color())
                axes[1].scatter(df_test_seed.epoch[start:], df_test_seed.fde[start:], 
                    c=p[-1].get_color(), s=1)
            axes[1].plot(df_test_seed.epoch[start:], test_fde[start:], 
                c=p[-1].get_color(), linewidth=2.5)

    axes[0].set_ylabel('ADE')
    axes[1].set_ylabel('FDE')
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    plt.savefig(out_path)   
    print('Saved', out_path)
    plt.close(fig)


def extract_curve_model(
    train_msgs, val_window=9,  
    show_train=False, show_raw_val=False, box_loc='middle',
    out_path='figures/training_curve/curve.png'):

    train_msgs_list = re.split('save_every_n', train_msgs)[1:]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for msg in train_msgs_list: 
        df_curve = extract_training_score(msg)

        experiment = re.search("Experiment (.*?) has started", msg).group(1)
        train_seed = int(re.search("'seed': ([\d+]),", msg).group(1))
        n_epoch = re.search("Best epoch at ([\d]+)", msg)
        n_epoch = int(n_epoch.group(1)) + 1 if n_epoch is not None else 199
        metric = re.search('Average performance \(by [\d]+\): \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)', msg)
        ade = round(float(metric.group(1)), 2)
        fde = round(float(metric.group(2)), 2)
        train_net = get_train_net(experiment)
        n_train = int(get_n_train(experiment))
        position = get_position(experiment, return_list=False)
        lr = get_lr(experiment)
        
        if position is not None:    
            label_name = f'TrS{train_seed}_{train_net}[{position}]({n_train})_{lr}_{ade}/{fde}'
        else:
            label_name = f'TrS{train_seed}_{train_net}({n_train})_{lr}_{ade}/{fde}'
        
        val_ade = df_curve.val_ade
        val_fde = df_curve.val_fde
        if val_window is not None:
            val_ade = moving_average(df_curve.val_ade, val_window, box_loc=box_loc)
            val_fde = moving_average(df_curve.val_fde, val_window, box_loc=box_loc)        

        start = 5
        # ade 
        # validation
        p = axes[0].plot(df_curve.epoch[start:], val_ade[start:], linewidth=1)
        axes[0].scatter(n_epoch, val_ade[n_epoch], c=p[-1].get_color(), marker='*')
        if show_raw_val:
            axes[0].plot(df_curve.epoch[start:], df_curve.val_ade[start:], 
                linewidth=0.5, alpha=0.5, c=p[-1].get_color())
            axes[0].scatter(df_curve.epoch[start:], df_curve.val_ade[start:], c=p[-1].get_color(), s=1)
        if show_train:
            axes[0].plot(df_curve.epoch[start:], df_curve.train_ade[start:], linestyle='--', c=p[-1].get_color())

        # fde 
        # validation 
        p = axes[1].plot(df_curve.epoch[start:], val_fde[start:], 
            label=label_name, linewidth=1)
        axes[1].scatter(n_epoch, val_fde[n_epoch], c=p[-1].get_color(), marker='*')
        if show_raw_val:
            axes[1].plot(df_curve.epoch[start:], df_curve.val_fde[start:], 
                linewidth=0.5, alpha=0.5, c=p[-1].get_color())
            axes[1].scatter(df_curve.epoch[start:], df_curve.val_fde[start:], 
                c=p[-1].get_color(), s=1)
        if show_train:
            axes[1].plot(df_curve.epoch[start:], df_curve.train_fde[start:], linestyle='--', c=p[-1].get_color())

    axes[0].set_ylabel('ADE')
    axes[1].set_ylabel('FDE')
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    plt.savefig(out_path)   
    print('Saved', out_path)
    plt.close(fig)


def extract_test_score(test_msg):
    msg_list = re.split('save_every_n', test_msg)[1:]
    df = pd.DataFrame(columns=['eval_seed', 'tuned_ckpt', 'ade', 'fde'])
    for msg in msg_list: 
        eval_seed = re.search("'seed': ([\d+]),", msg)
        metric = re.search('Average performance \(by [\d]+\): \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)', msg)
        tuned_ckpt = re.search("'tuned_ckpt': '(.*?)',", msg)

        df = pd.concat([df, pd.DataFrame({
            'eval_seed': eval_seed.group(1) if eval_seed is not None else None,
            'tuned_ckpt': tuned_ckpt.group(1).split('/')[-1] if tuned_ckpt is not None else None,
            'ade': metric.group(1) if metric is not None else None, 
            'fde': metric.group(2) if metric is not None else None}, index=[0])], )
    df.eval_seed = df.eval_seed.astype(int)
    df.ade = df.ade.astype(float)
    df.fde = df.fde.astype(float)
    df['train_seed'] = df['tuned_ckpt'].apply(lambda x: get_train_seed(x)).astype(int)
    df['epoch'] = df['tuned_ckpt'].apply(lambda x: get_epoch_number(x)).astype(int)
    # reorder columns 
    reordered_cols = ['train_seed', 'eval_seed', 'epoch', 'ade', 'fde', 'tuned_ckpt']
    df = df.reindex(columns=reordered_cols)
    return df


def get_epoch_number(ckpt_path):
    if ckpt_path is not None:
        return int(ckpt_path.split('__')[-1].split('.')[0].split('_')[-1])
    else:
        return None 


def get_train_seed(ckpt_path):
    if ckpt_path is not None:
        return int(ckpt_path.split('/')[-1].split('__')[0].split('_')[-1])
    else:
        return None 


def get_train_net(ckpt_path):
    if ckpt_path is not None:
        return ckpt_path.split('__')[2]
    else:
        return None 


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


def extract_file(
    file_path, test_file_path, out_dir, 
    val_window, test_window, box_loc, 
    show_train, show_raw_val, show_raw_test, diff):
    with open(file_path, 'r') as f:
        msgs = f.read()
    if test_file_path is not None:
        with open(test_file_path, 'r') as f:
            test_msgs = f.read()
    else:
        test_msgs = None 

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    if '/' in file_path:
        file_name = re.search('/([^/]+).out', file_path).group(1)
    else:
        file_name = file_path.replace('.out', '')

    if diff == 'seed':
        if test_file_path is None:
            out_path = f'{out_dir}/{file_name}__{box_loc}_{val_window}_{test_window}_{show_raw_val}_{show_raw_test}.png'  
        else:
            out_path = f'{out_dir}/{file_name}__test_{box_loc}_{val_window}_{test_window}_{show_raw_val}_{show_raw_test}.png'
        extract_curve_seed(train_msgs=msgs, test_msgs=test_msgs, 
            val_window=val_window, test_window=test_window, out_path=out_path, 
            box_loc=box_loc, show_raw_val=show_raw_val, show_raw_test=show_raw_test)
    elif diff == 'model':
        out_path = f'{out_dir}/{file_name}_{box_loc}_{val_window}_{show_train}_{show_raw_val}_{show_raw_test}.png'
        extract_curve_model(train_msgs=msgs, val_window=val_window, out_path=out_path, 
            box_loc=box_loc, show_raw_val=show_raw_val, show_train=show_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default=None, type=str)
    parser.add_argument('--test_file_path', default=None, type=str)
    parser.add_argument('--box_loc', default='middle', type=str, choices=['history', 'middle'])
    parser.add_argument('--val_window', default=9, type=int)
    parser.add_argument('--test_window', default=9, type=int)

    parser.add_argument('--show_train', action='store_true')
    parser.add_argument('--show_raw_val', action='store_true')
    parser.add_argument('--show_raw_test', action='store_true')
    parser.add_argument('--out_dir', default='./', type=str)
    parser.add_argument('--diff', choices=['seed', 'model'])
    args = parser.parse_args()
    
    extract_file(args.file_path, args.test_file_path, args.out_dir, 
        args.val_window, args.test_window, args.box_loc, 
        args.show_train, args.show_raw_val, args.show_raw_test, args.diff)
