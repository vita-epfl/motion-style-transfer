import os
import yaml
import time
import pathlib
import argparse
import numpy as np
import pandas as pd

from utils.data_utils import set_random_seeds, prepare_dataeset, get_meta_ids_focus
from utils.parser import get_parser
from utils.util import get_params, get_image_and_data_path, restore_model, get_ckpts_and_names
from evaluator.visualization import plot_prediction, plot_multiple_predictions, plot_goal_map_with_samples


def main(args):
    # configuration
    set_random_seeds(args.seed)
    params = get_params(args)
    obs_len = params['obs_len']
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # prepare data 
    _, _, df_test = prepare_dataeset(
        DATA_PATH, args.load_data, args.batch_size, None, 
        None, args.val_files, args.val_split, args.test_splits, 
        args.shuffle, args.share_val_test, 'eval', show_details=False)
    # get focused data 
    meta_ids_focus = get_meta_ids_focus(df_test, 
        given_csv={'path': args.result_path, 'name': args.result_name, 'n_limited': args.result_limited}, 
        given_meta_ids=args.given_meta_ids, random_n=args.random_n)
    df_test = df_test[df_test.metaId.isin(meta_ids_focus)]
    print('meta_ids_focus: #=', len(meta_ids_focus))
    print(f"df_test_limited: {df_test.shape}; #={df_test.metaId.unique().shape[0]}")
    
    # ckpts
    ckpts, ckpts_name, is_file_separated = get_ckpts_and_names(
        args.ckpts, args.ckpts_name, args.pretrained_ckpt, args.tuned_ckpts)

    # main  
    ckpts_trajs_dict = dict()
    for i, (ckpt, ckpt_name) in enumerate(zip(ckpts, ckpts_name)):
        print(f'====== Testing for {ckpt_name} ======')

        # load model
        model = restore_model(params, is_file_separated[i], 
            ckpt if not is_file_separated[i] else ckpts[0], 
            None if not is_file_separated[i] else ckpt)

        # test 
        set_random_seeds(args.seed)
        _, _, list_metrics, list_trajs = model.test(df_test, IMAGE_PATH, True, False) 

        # store ade/fde comparison
        for r in range(args.n_round):
            if r == 0:
                df_to_merge = list_metrics[r] 
            else:
                df_to_merge[['ade', 'fde']] = list_metrics[r][['ade', 'fde']] + df_to_merge[['ade', 'fde']]
        df_to_merge[['ade', 'fde']] = df_to_merge[['ade', 'fde']] / args.n_round
        df_to_merge = df_to_merge.rename({
            'ade': f'ade_{ckpt_name}', 'fde': f'fde_{ckpt_name}'}, axis=1)
        df_result = df_to_merge if i == 0 else df_result.merge(df_to_merge, on=['metaId', 'sceneId'])
        if args.n_round == 1:
            ckpts_trajs_dict[ckpt_name] = list_trajs[0]
        else:
            ckpts_trajs_dict[ckpt_name] = list_trajs

    if args.val_files is not None:
        folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.val_files).rstrip('.pkl')}" 
    else: 
        folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}"
    out_dir = f'csv/comparison/{folder_name}'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_name = f"{'_'.join(ckpts_name)}__N{len(meta_ids_focus)}_R{args.n_round}"
    out_name = f'{out_dir}/{csv_name}.csv'
    df_result.to_csv(out_name, index=False)
    print(f'Saved {out_name}')

    if args.viz:
        if args.n_round == 1:
            plot_prediction(IMAGE_PATH, ckpts_trajs_dict, 
                f'figures/prediction/{folder_name}/{"_".join(ckpts_name)}')
        else:
            plot_multiple_predictions(IMAGE_PATH, ckpts_trajs_dict, 
                f'figures/prediction_multiple/{folder_name}/{"_".join(ckpts_name)}', 
                obs_len=obs_len)
            # plot_goal_map_with_samples(IMAGE_PATH, ckpts_trajs_dict, 
            #     f'figures/goal_map_with_samples/{folder_name}')
        

def hook_store_output(module, input, output): 
    module.output = output


if __name__ == '__main__':
    parser = get_parser(False)
    # data
    parser.add_argument('--given_meta_ids', default=None, type=int, nargs='+')
    parser.add_argument('--result_path', default=None, type=str)
    parser.add_argument('--result_name', default=None, type=str)
    parser.add_argument('--result_limited', default=None, type=int)
    parser.add_argument('--random_n', default=None, type=int)
    parser.add_argument('--viz', action='store_true')

    args=parser.parse_args()

    main(args)

# python -m evaluator.evaluate_multickpts --config_filename inD_shortterm_eval.yaml --dataset_path filter/shortterm/agent_type/scene1/pedestrian_filter_s1_t524 --load_data predefined --network original --pretrained_ckpt ckpts/sdd__ynet__ped.pt --tuned_ckpts ckpts/inD/sdd_to_inD/Seed_1__filter_shortterm_agent_type_scene1_pedestrian_filter_s1_t524__mosa_2__Pos_0_1_2_3_4__TrN_20__lr_0.001__smooth__original.pt ckpts/inD/sdd_to_inD/Seed_1__filter_shortterm_agent_type_scene1_pedestrian_filter_s1_t524__all__TrN_20__lr_0.00005__original.pt --n_round 3 --viz