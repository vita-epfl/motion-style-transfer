import os
import time
import numpy as np 
import pandas as pd

from models.trainer import YNetTrainer
from utils.parser import get_parser
from utils.util import get_experiment_name, get_params, get_image_and_data_path
from utils.data_utils import set_random_seeds, prepare_dataeset
from evaluator.visualization import plot_given_trajectories_scenes_overlay


def main(args):
    # config 
    tic = time.time()
    set_random_seeds(args.seed)
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # load data 
    df_train, df_val, df_test = prepare_dataeset(
        DATA_PATH, args.load_data, args.batch_size, args.n_train_batch, 
        args.train_files, args.val_files, args.val_split, args.test_splits, 
        args.shuffle, args.share_val_test, 'train', args.show_details)

    folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}"

    # experiment name 
    EXPERIMENT_NAME = get_experiment_name(args, df_train.metaId.unique().shape[0])
    print(f"Experiment {EXPERIMENT_NAME} has started")

    # plot
    # plot_given_trajectories_scenes_overlay(IMAGE_PATH, df_train, f'figures/traj_check/{EXPERIMENT_NAME}/train')
    # plot_given_trajectories_scenes_overlay(IMAGE_PATH, df_val, f'figures/traj_check/{EXPERIMENT_NAME}/val')
    # plot_given_trajectories_scenes_overlay(IMAGE_PATH, df_test, f'figures/traj_check/{EXPERIMENT_NAME}/test')

    # model
    model = YNetTrainer(params=params)
    if args.pretrained_ckpt is not None:
        model.load_params(args.pretrained_ckpt)
        print(f"Loaded checkpoint {args.pretrained_ckpt}")
    else:
        print("Training from scratch")

    # initialization check 
    if args.init_check:
        params_pretrained = params.copy()
        params_pretrained.update({'position': []})
        pretrained_model = YNetTrainer(params=params_pretrained)
        pretrained_model.load_params(args.pretrained_ckpt)
        set_random_seeds(args.seed)
        ade_pre, fde_pre, _, _ = pretrained_model.test(df_test, IMAGE_PATH)
        set_random_seeds(args.seed)
        ade_cur, fde_cur, _, _ = model.test(df_test, IMAGE_PATH)
        if ade_pre != ade_cur or fde_pre != fde_cur:
            raise RuntimeError('Wrong model initialization')
        else:
            print('Passed initialization check')

    # training
    print('############ Train model ##############')
    val_ade, val_fde = model.train(df_train, df_val, IMAGE_PATH, IMAGE_PATH, EXPERIMENT_NAME)

    # test for leftout data 
    print('############ Test leftout data ##############')
    set_random_seeds(args.seed)
    test_ade, test_fde, _, _ = model.test(df_test, IMAGE_PATH)

    toc = time.time()
    print('Time spent:', time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))


if __name__ == '__main__':
    parser = get_parser(True)
    args = parser.parse_args()

    main(args)
