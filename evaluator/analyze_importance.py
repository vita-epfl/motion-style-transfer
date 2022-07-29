import os
import yaml
import pathlib
import argparse
from models.trainer import YNetTrainer
from utils.data_utils import set_random_seeds, dataset_split, dataset_given_scenes
from evaluator.visualization import plot_importance_analysis


def main(args):
    # ## configuration
    with open(os.path.join('config', 'sdd_raw_eval.yaml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params['segmentation_model_fp'] = os.path.join(params['data_dir'], params['dataset_name'], 'sdd_segmentation.pth')
    params.update(vars(args))
    print(params)

    # ## set up data 
    print('############ Prepare dataset ##############')
    # image path 
    IMAGE_PATH = os.path.join(params['data_dir'], params['dataset_name'], 'raw', 'annotations')
    assert os.path.isdir(IMAGE_PATH), 'raw data dir error'
    # data path 
    DATA_PATH = os.path.join(params['data_dir'], params['dataset_name'], args.dataset_path)

    # data 
    if args.n_leftouts:
        _, _, df_test = dataset_split(DATA_PATH, args.files, 0, args.n_leftouts)
    elif args.scenes:
        df_test = dataset_given_scenes(DATA_PATH, args.files, args.scenes)
    else:
        _, df_test, _ = dataset_split(DATA_PATH, args.files, 0)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # dir 
    folder_name = f"{args.seed}__{args.pretrained.split('_filter_')[1].split('__')[0]}__{'_'.join(args.files).rstrip('.pkl')}" 
    if args.replace_tuned:
        out_dir_tuned = f"csv/importance_analysis/{folder_name}/{args.depth}/replace_tuned"
        pathlib.Path(out_dir_tuned).mkdir(parents=True, exist_ok=True)
    if args.replace_pretrained:
        out_dir_pretrained = f"csv/importance_analysis/{folder_name}/{args.depth}/replace_pretrained"
        pathlib.Path(out_dir_pretrained).mkdir(parents=True, exist_ok=True)
        
    # ## model 
    if args.generate_csv:
        # pretrained model 
        pretrained_model = YNetTrainer(params=params)
        pretrained_model.load(args.pretrained)
        set_random_seeds(args.seed)
        _, _, list_metrics_pretrained, _ = pretrained_model.test(df_test, IMAGE_PATH, False, False) 
        if args.replace_tuned:
            list_metrics_pretrained[0].to_csv(
                f"{out_dir_tuned}/OODG__N{'_'.join(str(n) for n in args.n_leftouts)}.csv", index=False)
        if args.replace_pretrained:
            list_metrics_pretrained[0].to_csv(
                f"{out_dir_pretrained}/OODG__N{'_'.join(str(n) for n in args.n_leftouts)}.csv", index=False)
        print('Saved pretrained predictions')

        if args.depth == -1:
            # tuned models 
            for tuned, tuned_name in zip(args.tuned, args.tuned_name):
                print(f'====== Testing for {tuned_name} ======')
                tuned_model = YNetTrainer(params=params)
                tuned_model.load(tuned)
                set_random_seeds(args.seed)
                _, _, list_metrics_tuned, _ = tuned_model.test(df_test, IMAGE_PATH, False, False)         
                
                # replace one layers in tuned model by pretrained model 
                if args.replace_tuned:
                    list_metrics_tuned[0].to_csv(
                        f"{out_dir_tuned}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}.csv", index=False)
                    for param_name, param in pretrained_model.model.named_parameters():
                        # if (tuned_name == 'ET' and param_name.startswith('encoder')) or \
                        #     (tuned_name == 'FT' and not param_name.startswith('semantic_segmentation')):
                        if not param_name.startswith('semantic_segmentation'):
                            tuned_model.load(tuned)
                            print(f'Replacing {param_name}')
                            tuned_model.model.load_state_dict({param_name: param}, strict=False)
                            # sanity check 
                            # for i in tuned_model.model.state_dict().keys(): print(i, (tuned_model.model.state_dict()[i] == pretrained_model.model.state_dict()[i]).all().item())
                            set_random_seeds(args.seed)
                            _, _, list_metrics, _ = tuned_model.test(df_test, IMAGE_PATH, False, False) 
                            # store ade/fde 
                            out_path = f"{out_dir_tuned}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}__{param_name}.csv"
                            list_metrics[0].to_csv(out_path, index=False)
                
                # replace one layer in pretrained model by tuned model 
                if args.replace_pretrained:
                    list_metrics_tuned[0].to_csv(
                        f"{out_dir_pretrained}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}.csv", index=False)
                    tuned_model.load(tuned)
                    for param_name, param in tuned_model.model.named_parameters():
                        if not param_name.startswith('semantic_segmentation'):
                            pretrained_model.load(args.pretrained)
                            print(f'Replacing {param_name}')
                            pretrained_model.model.load_state_dict({param_name: param}, strict=False)
                            set_random_seeds(args.seed)
                            _, _, list_metrics, _ = pretrained_model.test(df_test, IMAGE_PATH, False, False) 
                            # store ade/fde 
                            out_path = f"{out_dir_pretrained}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}__{param_name}.csv"
                            list_metrics[0].to_csv(out_path, index=False)
        
        elif (args.depth == 1) | (args.depth == 2):
            # group parameters name 
            state_dict = pretrained_model.model.state_dict().keys()
            params_selected = []
            # encoder
            for param in state_dict:
                if param.startswith('encoder'):
                    if args.depth == 1:
                        p_short = '.'.join(param.split('.')[:-2])
                    elif args.depth == 2:
                        p_short = '.'.join(param.split('.')[:-1])
                    if p_short not in params_selected:
                        params_selected.append(p_short)
            # decoder
            for decoder in ['goal_decoder', 'traj_decoder']:
                if args.depth == 1:
                    params_selected.append(f'{decoder}.center')
                    for i in range(5):
                        params_selected.append(
                            [f'{decoder}.upsample_conv.{i}', f'{decoder}.decoder.{i}'])
                    params_selected.append(f'{decoder}.predictor')
                elif args.depth == 2:
                    for param in state_dict:
                        if param.startswith(decoder):
                            p_short = '.'.join(param.split('.')[:-1])
                            if p_short not in params_selected:
                                params_selected.append(p_short)

            # tuned models 
            for tuned, tuned_name in zip(args.tuned, args.tuned_name):
                print(f'====== Testing for {tuned_name} ======')
                tuned_model = YNetTrainer(params=params)
                tuned_model.load(tuned)
                set_random_seeds(args.seed)
                _, _, list_metrics_tuned, _ = tuned_model.test(df_test, IMAGE_PATH, False, False)         
                
                if args.replace_tuned:
                    list_metrics_tuned[0].to_csv(
                        f"{out_dir_tuned}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}.csv", index=False)
                    for param_selected in params_selected:
                        tuned_model.load(tuned)
                        for param_name, param in pretrained_model.model.named_parameters():
                            if not param_name.startswith('semantic_segmentation'):
                                # "semantic_segmentation".num_batches is changed
                                replace = False
                                if isinstance(param_selected, str):
                                    if param_selected in param_name:
                                        replace = True
                                if isinstance(param_selected, list):
                                    if any([p for p in param_selected if p in param_name]):
                                        replace = True
                                if replace:        
                                    print(f'Replacing {param_name}')
                                    tuned_model.model.load_state_dict({param_name: param}, strict=False)
                        set_random_seeds(args.seed)
                        _, _, list_metrics, _ = tuned_model.test(df_test, IMAGE_PATH, False, False) 
                        # store ade/fde 
                        out_path = f"{out_dir_tuned}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}__{param_selected}.csv"
                        list_metrics[0].to_csv(out_path, index=False)
                        print(f'Saved {out_path}')
                
                # replace one layer in pretrained model by tuned model 
                if args.replace_pretrained:
                    list_metrics_tuned[0].to_csv(
                        f"{out_dir_pretrained}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}.csv", index=False)
                    tuned_model.load(tuned)
                    for param_selected in params_selected:
                        pretrained_model.load(args.pretrained)
                        for param_name, param in tuned_model.model.named_parameters():
                            if not param_name.startswith('semantic_segmentation'):
                                replace = False
                                if isinstance(param_selected, str):
                                    if param_selected in param_name:
                                        replace = True
                                if isinstance(param_selected, list):
                                    if any([p for p in param_selected if p in param_name]):
                                        replace = True
                                if replace:        
                                    print(f'Replacing {param_name}')
                                    pretrained_model.model.load_state_dict({param_name: param}, strict=False)
                        set_random_seeds(args.seed)
                        _, _, list_metrics, _ = pretrained_model.test(df_test, IMAGE_PATH, False, False) 
                        # store ade/fde 
                        out_path = f"{out_dir_pretrained}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}__{param_selected}.csv"
                        list_metrics[0].to_csv(out_path, index=False) 
                        print(f'Saved {out_path}')
        else:
            raise ValueError(f'No support for depth={args.depth}')

    # visualize
    if args.replace_tuned:
        plot_importance_analysis(
            f'csv/importance_analysis/{folder_name}/{args.depth}/replace_tuned', 
            f'figures/importance_analysis/{folder_name}/{args.depth}/replace_tuned',
            n_test=args.n_leftouts[0], depth=args.depth)
    if args.replace_pretrained:
        plot_importance_analysis(
            f'csv/importance_analysis/{folder_name}/{args.depth}/replace_pretrained', 
            f'figures/importance_analysis/{folder_name}/{args.depth}/replace_pretrained',
            n_test=args.n_leftouts[0], depth=args.depth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--dataset_name', default='sdd', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_round', default=1, type=int)
    # files 
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--tuned', default=None, type=str, nargs='+')
    parser.add_argument('--tuned_name', default=None, type=str, nargs='+')
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--files', default=None, type=str, nargs='+')
    parser.add_argument('--n_leftouts', default=None, type=int, nargs='+')
    parser.add_argument('--scenes', default=None, type=str, nargs='+')
    parser.add_argument('--depth', default=-1, type=int)
    parser.add_argument('--replace_tuned', action='store_true')
    parser.add_argument('--replace_pretrained', action='store_true')
    parser.add_argument('--generate_csv', action='store_true')

    args=parser.parse_args()

    main(args)
