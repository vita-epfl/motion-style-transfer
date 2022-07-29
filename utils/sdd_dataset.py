import os
import argparse
import numpy as np
import pandas as pd

from utils.data_utils import split_fragmented, downsample, \
    filter_short_trajectories, sliding_window, get_varf_table, \
	create_dataset_given_range, create_dataset_by_agent_type, compute_distance_with_neighbors


def load_raw_sdd(path):
    data_path = os.path.join(path, "annotations")
    scenes_main = os.listdir(data_path)
    SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax',
                'frame', 'lost', 'occluded', 'generated', 'label']
    data = []
    for scene_main in sorted(scenes_main):
        scene_main_path = os.path.join(data_path, scene_main)
        for scene_sub in sorted(os.listdir(scene_main_path)):
            scene_path = os.path.join(scene_main_path, scene_sub)
            annot_path = os.path.join(scene_path, 'annotations.txt')
            scene_df = pd.read_csv(annot_path, header=0,
                                   names=SDD_cols, delimiter=' ')
            # Calculate center point of bounding box
            scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
            scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
            scene_df = scene_df[scene_df['lost'] == 0]  # drop lost samples
            scene_df = scene_df.drop(
                columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'lost'])
            scene_df['sceneId'] = f"{scene_main}_{scene_sub.split('video')[1]}"
            # new unique id by combining scene_id and track_id
            scene_df['rec&trackId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in
                                       zip(scene_df.sceneId, scene_df.trackId)]
            data.append(scene_df)
    data = pd.concat(data, ignore_index=True)
    rec_trackId2metaId = {}
    for i, j in enumerate(data['rec&trackId'].unique()):
        rec_trackId2metaId[j] = i
    data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
    data = data.drop(columns=['rec&trackId'])
    return data


def load_and_window_sdd(path, step, window_size, stride):
    df = load_raw_sdd(path=path)
    df = split_fragmented(df)  # split track if frame is not continuous
    df = downsample(df, step=step)
    df = filter_short_trajectories(df, threshold=window_size)
    df = sliding_window(df, window_size=window_size, stride=stride)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--additional_data_dir', default='data/sdd/raw', type=str, 
        help='Path to the scene images and variation factor file')
    parser.add_argument('--raw_data_dir', default='data/sdd/raw', type=str, 
        help='Path to the raw data, can be a subset of the entire dataset')
    parser.add_argument('--raw_data_filename', default='data_8_12_2_5fps.pkl', type=str)
    parser.add_argument('--filter_data_dir', default='data/sdd/filter/shortterm', type=str)

    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--statistic_only', action='store_true', 
        help='By default, show the statistics and save the customized dataset. ' + \
            'Set False to show only the statistics of data split.')
    
    parser.add_argument("--step", default=12, type=int)
    parser.add_argument("--window_size", default=20, type=int)
    parser.add_argument("--stride", default=20, type=int)
    parser.add_argument("--obs_len", default=8, type=int)

    parser.add_argument("--varf", default=None, nargs='+',
                        help="Variation factors from: 'avg_vel', 'max_vel', "+\
                            "'avg_acc', 'max_acc', 'abs+max_acc', 'abs+avg_acc', "+\
                            "'min_dist', 'avg_den50', 'avg_den100', 'agent_type'")
    parser.add_argument("--varf_ranges", help='range of varation factor to take', 
                        default=[(0.5, 3.5), (4, 8)])

    parser.add_argument("--labels", default=['Pedestrian', 'Biker'], nargs='+', type=str,
                        choices=['Biker', 'Bus', 'Car', 'Cart', 'Pedestrian', 'Skater'])
    parser.add_argument('--selected_scenes', default=None, type=str, nargs='+')

    args = parser.parse_args()
    args.labels.sort()
    print(args)

    # ============== load raw dataset ===============
    if not args.reload:
        # ## load raw dataset
        df = load_and_window_sdd(args.raw_data_dir, args.step, args.window_size, args.stride)
        print('Loaded raw dataset')
        # possibly add a column of distance with neighbors 
        if args.varf is not None:
            if np.array(['dist' in f or 'den' in f for f in args.varf]).any():
                out = df.groupby('sceneId').apply(compute_distance_with_neighbors)
                for idx_1st in out.index.get_level_values('sceneId').unique():
                    df.loc[out[idx_1st].index, 'dist'] = out[idx_1st].values
                print(f'Added a column of distance with neighbors to df')
        out_path = os.path.join(args.raw_data_dir, args.raw_data_filename)
        df.to_pickle(out_path)
        print(f'Saved data to {out_path}')

        # ## get variation factor table 
        varf_list = ['avg_vel', 'max_acc']
        df_varf = get_varf_table(df, varf_list, args.obs_len)
        out_path = os.path.join(args.additional_data_dir, args.raw_data_filename.replace('data', 'varf'))
        df_varf.to_pickle(out_path)
        print(f'Saved variation factor data to {out_path}')

    else:  # reload = True
        # ## or load from stored pickle
        df = pd.read_pickle(os.path.join(args.raw_data_dir, args.raw_data_filename))
        print('Reloaded raw dataset')

    # ============== create customized dataset ================
    if args.varf is not None:
        if args.varf == ['agent_type']:
            out_dir = os.path.join(args.filter_data_dir, args.varf[0])
            create_dataset_by_agent_type(df, args.labels, out_dir, 
                statistic_only=args.statistic_only, selected_scenes=args.selected_scenes)
        else:
            out_dir = os.path.join(args.filter_data_dir, '__'.join(args.varf), '_'.join(args.labels))
            create_dataset_given_range(df, args.varf, args.varf_ranges, args.labels, 
                out_dir, obs_len=args.obs_len, statistic_only=args.statistic_only)
        print(f'Created dataset: \nVariation factor = {args.varf} \nAgents = {args.labels}')
