import time 
import numpy as np
from utils.parser import get_parser
from utils.data_utils import set_random_seeds, prepare_dataeset
from utils.util import get_params, get_image_and_data_path, restore_model, get_ckpts_and_names
from evaluator.visualization import plot_given_trajectories_scenes_overlay


def main(args):
    # configuration
    tic = time.time()
    set_random_seeds(args.seed)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # prepare data 
    _, _, df_test = prepare_dataeset(DATA_PATH, args.load_data, args.batch_size, 
        None, None, args.val_files, args.val_split, args.test_splits, 
        args.shuffle, args.share_val_test, 'eval', args.show_details)
    
    # if args.pretrained_ckpt is not None:
    #     plot_given_trajectories_scenes_overlay(IMAGE_PATH, df_test, f'figures/traj_check/{args.tuned_ckpt}/test')
    # else:
    #     plot_given_trajectories_scenes_overlay(IMAGE_PATH, df_test, f'figures/traj_check/{args.ckpts}/test')

    # ckpts
    ckpts, ckpts_name, is_file_separated = get_ckpts_and_names(
        args.ckpts, args.ckpts_name, args.pretrained_ckpt, [args.tuned_ckpt])
    print(ckpts, ckpts_name)
    if len(ckpts_name) == 1:
        model = restore_model(params, is_file_separated[0],
            ckpts[0] if not is_file_separated[0] else args.pretrained_ckpt, 
            None if not is_file_separated[0] else ckpts[0])
    elif len(ckpts_name) > 1:
        for i, (ckpt, ckpt_name) in enumerate(zip(ckpts, ckpts_name)):
            if ckpt_name != 'OODG':
                model = restore_model(params, is_file_separated[i], 
                    ckpt if not is_file_separated[i] else ckpts[0], 
                    None if not is_file_separated[i] else ckpt)
    
    # test
    print('############ Test model ##############')
    set_random_seeds(args.seed)
    ade, fde, _, _ = model.test(df_test, IMAGE_PATH)

    toc = time.time()
    print('Time spent:', time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))


if __name__ == '__main__':
    parser = get_parser(False)
    args = parser.parse_args()
    main(args)
