import pandas as pd
from utils.parser import get_parser
from utils.data_utils import set_random_seeds, prepare_dataeset, get_meta_ids_focus
from utils.util import get_params, get_image_and_data_path, restore_model, get_ckpts_and_names
from evaluator.visualization import plot_goal_output, plot_activation


def main(args):
    # configuration
    set_random_seeds(args.seed)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)
    params.update({'decision': 'map'})

    # prepare data 
    _, _, df_test = prepare_dataeset(
        DATA_PATH, args.load_data, args.batch_size, None, 
        None, args.val_files, args.val_split, args.test_splits, 
        args.shuffle, args.share_val_test, 'eval', show_details=False)
    meta_ids_focus = get_meta_ids_focus(df_test, 
        given_csv={'path': args.result_path, 'name': args.result_name, 'n_limited': args.result_limited}, 
        given_meta_ids=args.given_meta_ids, random_n=args.random_n)
    df_test = df_test[df_test.metaId.isin(meta_ids_focus)]
    print(f"df_test_limited: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ckpts
    ckpts, ckpts_name, is_file_separated = get_ckpts_and_names(
        args.ckpts, args.ckpts_name, args.pretrained_ckpt, args.tuned_ckpts)

    # main  
    ckpts_hook_dict = {}
    for i, (ckpt, ckpt_name) in enumerate(zip(ckpts, ckpts_name)):
        ckpts_hook_dict[ckpt_name] = {}
        print(f'====== {ckpt_name} ======')

        # model 
        model = restore_model(params, is_file_separated[i], 
            ckpt if not is_file_separated[i] else ckpts[0], 
            None if not is_file_separated[i] else ckpt)
        model.model.eval()

        # register layer 
        # layers_dict = {}
        # layers_dict = {
        #     'encoder.stages.0.0': model.model.encoder.stages[0][0],
        #     'encoder.stages.0.1': model.model.encoder.stages[0][1],
        #     'encoder.stages.1.4': model.model.encoder.stages[1][4],
        #     'encoder.stages.2.4': model.model.encoder.stages[2][4], 
        #     'encoder.stages.3.4': model.model.encoder.stages[3][4], 
        #     'encoder.stages.4.4': model.model.encoder.stages[4][4],
        #     'encoder.stages.5.0': model.model.encoder.stages[5][0],
        # }
        # layers_hook = {}
        # for layer_name, layer in layers_dict.items():
        #     layer.register_forward_hook(hook_store_output)
        #     layers_hook[layer_name] = layer
        # layer = model.model.encoder.scene_stages[0][0]
        # layer.register_forward_hook(hook_store_input)
        # layers_hook['encoder.scene_stages.0.0_input'] = layer

        # layer = model.model.encoder.motion_stages[0][0]
        # layer.register_forward_hook(hook_store_input)
        # layers_hook['encoder.motion_stages.0.0_input'] = layer
            
        # forward 
        pred_goal_map, pred_traj_map, raw_img = model.forward_test(
            df_test, IMAGE_PATH, set_input=[], noisy_std_frac=None) 
        
        # store 
        # for layer_name in layers_dict.keys():
        #     ckpts_hook_dict[ckpt_name][layer_name+'_output'] = layers_hook[layer_name].output
        # ckpts_hook_dict[ckpt_name]['encoder.scene_stages.0.0_input'] = \
        #     layers_hook['encoder.scene_stages.0.0_input'].input
        # ckpts_hook_dict[ckpt_name]['encoder.motion_stages.0.0_input'] = \
        #     layers_hook['encoder.motion_stages.0.0_input'].input
        # semantic_imgs = layers_hook['encoder.stages.0.0_input'].input[:, :6]
        pred_goal_map_sigmoid = model.model.sigmoid(pred_goal_map / params['temperature'])  
        pred_traj_map_sigmoid = model.model.sigmoid(pred_traj_map / params['temperature'])
        ckpts_hook_dict[ckpt_name]['goal_decoder.predictor_sigmoid'] = pred_goal_map_sigmoid
        ckpts_hook_dict[ckpt_name]['traj_decoder.predictor_sigmoid'] = pred_traj_map_sigmoid
        # ckpts_hook_dict[ckpt_name]['goal_decoder.predictor_output'] = pred_goal_map
        # ckpts_hook_dict[ckpt_name]['traj_decoder.predictor_output'] = pred_traj_map
        
        for key, value in ckpts_hook_dict[ckpt_name].items():
            print(key, value.shape)
        # print('semantic_imgs', semantic_imgs.shape)

    # plot 
    if args.val_files is not None:
        folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.val_files).rstrip('.pkl')}/{'_'.join(ckpts_name)}" 
    else:
        folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}/{'_'.join(ckpts_name)}" 
    index = df_test.groupby(by=['metaId', 'sceneId']).count().index
    print(index)
    # plot_goal_output(
    #     ckpts_hook_dict, index, df_test, IMAGE_PATH, 
    #     out_dir='figures/activation_yy', display_scene_img=False,
    #     inhance_threshold=0.1, format='png')
    # plot_goal_output(
    #     ckpts_hook_dict, index, df_test, IMAGE_PATH, 
    #     out_dir='figures/activation_yy', display_scene_img=False,
    #     inhance_threshold=0.05, format='png')
    plot_goal_output(
        ckpts_hook_dict, index, df_test, IMAGE_PATH, 
        out_dir='figures/activation_yy/ped2ped_new/', display_scene_img=False,
        inhance_threshold=0.1, format='pdf', white_bg=True, window=[650, 400, 450, 0.75, 'landscape'])
    # plot_goal_output(
    #     ckpts_hook_dict, index, df_test, IMAGE_PATH, 
    #     out_dir='figures/activation_yy', display_scene_img=False,
    #     inhance_threshold=0.15, format='pdf')
    # plot_goal_output(
    #     ckpts_hook_dict, index, df_test, IMAGE_PATH, 
    #     out_dir='figures/activation_yy', display_scene_img=True,
    #     inhance_threshold=0.1, zoom_in=False)
    # plot_goal_output(
    #     ckpts_hook_dict, index, df_test, IMAGE_PATH, 
    #     out_dir='figures/activation_yy', display_scene_img=True,
    #     inhance_threshold=0.15, zoom_in=False)
    # plot_activation(ckpts_hook_dict, index, 
    #     f'figures/activation_yy/{folder_name}', 
    #     compare_raw=args.compare_raw, compare_diff=args.compare_diff, 
    #     compare_overlay=args.compare_overlay, compare_relative=args.compare_relative, 
    #     scene_imgs=raw_img, semantic_imgs=None, scale_row=True, inhance_diff=False)
    # plot_activation(ckpts_hook_dict, index, 
    #     f'figures/activation/{folder_name}', 
    #     compare_raw=args.compare_raw, compare_diff=args.compare_diff, 
    #     compare_overlay=args.compare_overlay, compare_relative=args.compare_relative,
    #     scene_imgs=raw_img, semantic_imgs=None, scale_row=False, inhance_diff=False)
    

def hook_store_input(module, input, output):
    module.input = input[0]


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
    # plot 
    parser.add_argument('--compare_raw', action='store_true')
    parser.add_argument('--compare_diff', action='store_true')
    parser.add_argument('--compare_overlay', action='store_true')
    parser.add_argument('--compare_relative', action='store_true')
    args = parser.parse_args()
    main(args)

# metaid = 339: window=[800, 420, 700, 0.618, 'landscape'] / [800, 420, 550, 0.75, 'landscape']
# metaid = 337: window=[800, 420, 700, 0.618, 'landscape'] / [800, 420, 550, 0.75, 'landscape']
# metaid = 623: window=[900, 280, 700, 0.618, 'landscape']
# metaid = 312: window=[580, 600, 550, 0.618, 'landscape']
# metaid = 370: window=[630, 550, 500, 0.618, 'landscape']
# metaid = 90: window=[650, 400, 500, 0.618, 'landscape']
# metaid = 81: window=[650, 400, 500, 0.618, 'landscape'] / [650, 400, 450, 0.75, 'landscape']

# python -m evaluator.visualize_activation --config_filename inD_longterm_eval.yaml --load_data predefined --network original --dataset_path filter/agent_type/scene1/pedestrian_30_third --pretrained_ckpt ckpts/inD_longterm_weights.pt --tuned_ckpts ckpts/Seed_2__filter_agent_type_scene1_pedestrian_30_third__mosa_1__Pos_0_1_2_3_4__TrN_30__lr_0.005.pt --given_meta_ids 339


# metaid = 4419: window=[920, 350, 400, 0.618, 'portrait'] / [920, 350, 400, 0.75, 'portrait']
# metaid = 2632: window=[825, 325, 400, 0.618, 'portrait'] / [825, 325, 350, 0.75, 'portrait']
# metaid = 5058: window=[950, 250, 400, 0.618, 'portrait'] / [950, 210, 370, 0.75, 'portrait']

# python -m evaluator.visualize_activation --config_filename inD_shortterm_eval.yaml --load_data predefined --network fusion --n_fusion 2 --dataset_path filter/agent_type/scene1/truck_bus_filter --pretrained_ckpt ckpts/inD__ynetmod__car.pt --tuned_ckpts ckpts/Seed_1__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_scene__TrN_20__lr_0.001.pt ckpts/Seed_1__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_motion__TrN_20__lr_0.001.pt ckpts/Seed_1__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_scene_fusion__TrN_20__lr_0.001.pt ckpts/Seed_1__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_motion_fusion__TrN_20__lr_0.001.pt ckpts/Seed_1__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_scene_motion_fusion__TrN_20__lr_0.001.pt ckpts/Seed_1__filter_agent_type_scene1_truck_bus_filter__all__TrN_20__lr_0.00005.pt --given_meta_ids 5058 2632

# python -m evaluator.visualize_activation --config_filename inD_shortterm_eval.yaml --load_data predefined --network fusion --n_fusion 2 --dataset_path filter/agent_type/scene1/truck_bus_filter --pretrained_ckpt ckpts/inD__ynetmod__car.pt --tuned_ckpts ckpts/Seed_9__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_scene__TrN_20__lr_0.001.pt ckpts/Seed_9__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_motion__TrN_20__lr_0.001.pt ckpts/Seed_9__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_scene_fusion__TrN_20__lr_0.001.pt ckpts/Seed_9__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_motion_fusion__TrN_20__lr_0.001.pt ckpts/Seed_9__filter_agent_type_scene1_truck_bus_filter__mosa_1__Pos_scene_motion_fusion__TrN_20__lr_0.001.pt ckpts/Seed_1__filter_agent_type_scene1_truck_bus_filter__all__TrN_20__lr_0.00005.pt --given_meta_ids 4419


# --given_meta_ids 1317 559 279 647 5403 3479 4419 3859

# --given_meta_ids 5358 5883 5982

# --given_meta_ids 5711 6060 5670 5334 6063 6269 5885 6310 5767 6322 5445 5358 5726 6230 5468 5890 5883 5715 5450 6228 

