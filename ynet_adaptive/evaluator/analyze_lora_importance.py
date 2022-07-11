import numpy as np 
import pandas as pd
import torch 
import pathlib
import itertools
from utils.parser import get_parser
from utils.data_utils import set_random_seeds, dataset_split, get_meta_ids_focus
from utils.util import get_params, get_image_and_data_path, restore_model, get_ckpts_and_names


def main(args):
    # configuration
    set_random_seeds(args.seed)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # prepare data 
    _, _, df_test = dataset_split(DATA_PATH, args.val_files, 0, args.n_leftouts)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")
    meta_ids_focus = get_meta_ids_focus(df_test, 
        given_csv={'path': args.result_path, 'name': args.result_name, 'n_limited': args.result_limited}, 
        given_meta_ids=args.given_meta_ids, random_n=args.random_n)
    df_test = df_test[df_test.metaId.isin(meta_ids_focus)]
    print(f"df_test_limited: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ckpts
    ckpts, ckpts_name, is_file_separated = get_ckpts_and_names(
        args.ckpts, args.ckpts_name, args.pretrained_ckpt, args.tuned_ckpts)
    
    # main  
    model = restore_model(params, is_file_separated, ckpts[0], ckpts[1])
    model.model.eval()

    # main 
    folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.val_files).rstrip('.pkl')}"
    out_dir_tuned = f'csv/comparison/{folder_name}/replace_tuned'
    out_dir_pretrained = f'csv/comparison/{folder_name}/replace_pretrained'
    pathlib.Path(out_dir_tuned).mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_dir_pretrained).mkdir(parents=True, exist_ok=True)
    if not args.combine_layers:
        if args.replace_tuned:
            for param_name, param in model.model.named_parameters():
                if 'lora_A' in param_name:
                    # prepare model 
                    print(f'Replacing {param_name}')
                    model_mod = restore_model(params, is_file_separated, ckpts[0], ckpts[1])
                    model_mod.model.load_state_dict({param_name: torch.zeros(param.shape)}, strict=False)
                    # test 
                    set_random_seeds(args.seed)
                    _, _, list_metrics, _ = model_mod.test(df_test, IMAGE_PATH, False, False) 
                    if args.store_csv:
                        # store ade/fde 
                        out_path = f"{out_dir_tuned}/{ckpts_name[1]}__N{len(meta_ids_focus)}__{param_name}.csv"
                        list_metrics[0].to_csv(out_path, index=False)
        if args.replace_pretrained:
            for param_name, param in model.model.named_parameters():
                if 'lora_A' in param_name:
                    # prepare model 
                    print(f'Replacing LoRA layers except {param_name}')
                    model_mod = restore_model(params, is_file_separated, ckpts[0], ckpts[1])
                    for n, p in model_mod.model.named_parameters():
                        if 'lora_A' in n and n != param_name:
                            model_mod.model.load_state_dict({n: torch.zeros(p.shape)}, strict=False)
                    # test 
                    set_random_seeds(args.seed)
                    _, _, list_metrics, _ = model_mod.test(df_test, IMAGE_PATH, False, False) 
                    if args.store_csv:
                        # store ade/fde 
                        out_path = f"{out_dir_pretrained}/{ckpts_name[1]}__N{len(meta_ids_focus)}__{param_name}.csv"
                        list_metrics[0].to_csv(out_path, index=False)
    else:
        # given combination 
        # layers_combination = [
        #     ['encoder.stages.0.0.lora_A', 'encoder.stages.1.3.lora_A'],
        #     ['encoder.stages.0.0.lora_A', 'encoder.stages.2.1.lora_A'],
        #     ['encoder.stages.0.0.lora_A', 'encoder.stages.4.3.lora_A'],
        #     ['encoder.stages.0.0.lora_A', 'encoder.stages.1.3.lora_A', 'encoder.stages.2.1.lora_A'],
        #     ['encoder.stages.0.0.lora_A', 'encoder.stages.1.3.lora_A', 'encoder.stages.4.3.lora_A'], 
        #     ['encoder.stages.0.0.lora_A', 'encoder.stages.2.1.lora_A', 'encoder.stages.4.3.lora_A']
        # ]
        # or given number of layers to consider 
        layers_combination = []
        lora_layers = [n for n, _ in model.model.named_parameters() if 'lora_A' in n]
        for i in [2 ,3]: layers_combination += [j for j in itertools.combinations(lora_layers, i) if 'encoder.stages.0.0.lora_A' in j]
        if args.replace_tuned:
            for given_layers in layers_combination:
                model_mod = restore_model(params, is_file_separated, ckpts[0], ckpts[1])
                print(f'Replacing {given_layers}')
                # prepare model 
                for param_name, param in model_mod.model.named_parameters():
                    if param_name in given_layers:
                        print(f'{param_name}')
                        model_mod.model.load_state_dict({param_name: torch.zeros(param.shape)}, strict=False)
                # test 
                set_random_seeds(args.seed)
                _, _, list_metrics, _ = model_mod.test(df_test, IMAGE_PATH, False, False) 
                if args.store_csv:
                    # store ade/fde 
                    out_path = f"{out_dir_tuned}/{ckpts_name[1]}__N{len(meta_ids_focus)}__{'_'.join(given_layers)}.csv"
                    list_metrics[0].to_csv(out_path, index=False)
        if args.replace_pretrained:
            for given_layers in layers_combination:
                model_mod = restore_model(params, is_file_separated, ckpts[0], ckpts[1])
                print(f'Replacing LoRA layers except {given_layers}')
                # prepare model 
                for param_name, param in model_mod.model.named_parameters():
                    if 'lora_A' in param_name and param_name not in given_layers:
                        model_mod.model.load_state_dict({param_name: torch.zeros(param.shape)}, strict=False)
                # test 
                set_random_seeds(args.seed)
                _, _, list_metrics, _ = model_mod.test(df_test, IMAGE_PATH, False, False) 
                if args.store_csv:
                    # store ade/fde 
                    out_path = f"{out_dir_pretrained}/{ckpts_name[1]}__N{len(meta_ids_focus)}__{'_'.join(given_layers)}.csv"
                    list_metrics[0].to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = get_parser(False)
    # data
    parser.add_argument('--given_meta_ids', default=None, type=int, nargs='+')
    parser.add_argument('--result_path', default=None, type=str)
    parser.add_argument('--result_name', default=None, type=str)
    parser.add_argument('--result_limited', default=None, type=int)
    parser.add_argument('--random_n', default=None, type=int)
    # importance 
    parser.add_argument('--replace_tuned', action='store_true')
    parser.add_argument('--replace_pretrained', action='store_true')
    parser.add_argument('--combine_layers', action='store_true')
    parser.add_argument('--store_csv', action='store_true')

    args = parser.parse_args()
    main(args)

# python -m pdb -m evaluator.analyze_lora_importance --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/DC0__lora/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__lora_1__Pos_0_1_2_3_4__TrN_20__lr_0.0005.pt --val_files Biker.pkl --n_leftouts 10 
