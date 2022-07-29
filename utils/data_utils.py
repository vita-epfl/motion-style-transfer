import os
import cv2
import torch
import random
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'figure.max_open_warning': 0})


def mask_step(x, step):
    """
    Create a mask to only contain the step-th element starting from the first element. Used to downsample
    """
    mask = np.zeros_like(x)
    mask[::step] = 1
    return mask.astype(bool)


def downsample(df, step):
    """
    Downsample data by the given step. Example, SDD is recorded in 30 fps, with step=30, the fps of the resulting
    df will become 1 fps. With step=12 the result will be 2.5 fps. It will do so individually for each unique
    pedestrian (metaId)
    :param df: pandas DataFrame - necessary to have column 'metaId'
    :param step: int - step size, similar to slicing-step param as in array[start:end:step]
    :return: pd.df - downsampled
    """
    mask = df.groupby(['metaId'])['metaId'].transform(mask_step, step=step)
    return df[mask]


def filter_short_trajectories(df, threshold):
    """
    Filter trajectories that are shorter in timesteps than the threshold
    :param df: pandas df with columns=['x', 'y', 'frame', 'trackId', 'sceneId', 'metaId']
    :param threshold: int - number of timesteps as threshold, only trajectories over threshold are kept
    :return: pd.df with trajectory length over threshold
    """
    len_per_id = df.groupby(by='metaId', as_index=False).count(
    )  # sequence-length for each unique pedestrian
    idx_over_thres = len_per_id[len_per_id['frame'] >= threshold]  # rows which are above threshold
    idx_over_thres = idx_over_thres['metaId'].unique() # only get metaIdx with sequence-length longer than threshold
    df = df[df['metaId'].isin(idx_over_thres)] # filter df to only contain long trajectories
    return df


def groupby_sliding_window(x, window_size, stride):
    x_len = len(x)
    n_chunk = (x_len - window_size) // stride + 1
    idx = []
    metaId = []
    for i in range(n_chunk):
        idx += list(range(i * stride, i * stride + window_size))
        metaId += ['{}_{}'.format(x.metaId.unique()[0], i)] * window_size
    df = x.iloc()[idx]
    df['newMetaId'] = metaId
    return df


def sliding_window(df, window_size, stride):
    """
    Assumes downsampled df, chunks trajectories into chunks of length window_size. When stride < window_size then
    chunked trajectories are overlapping
    :param df: df
    :param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
    :param stride: timesteps to move from one trajectory to the next one
    :return: df with chunked trajectories
    """
    gb = df.groupby(['metaId'], as_index=False)
    df = gb.apply(groupby_sliding_window, window_size=window_size, stride=stride)
    df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
    df = df.drop(columns='newMetaId')
    df = df.reset_index(drop=True)
    return df


def split_at_fragment_lambda(x, frag_idx, gb_frag):
    """ Used only for split_fragmented() """
    metaId = x.metaId.iloc()[0]
    counter = 0
    if metaId in frag_idx:
        split_idx = gb_frag.groups[metaId]
        for split_id in split_idx:
            x.loc[split_id:, 'newMetaId'] = '{}_{}'.format(metaId, counter)
            counter += 1
    return x


def split_fragmented(df):
    """
    Split trajectories when fragmented (defined as frame_{t+1} - frame_{t} > 1)
    Formally, this is done by changing the metaId at the fragmented frame and below
    :param df: DataFrame containing trajectories
    :return: df: DataFrame containing trajectories without fragments
    """

    gb = df.groupby('metaId', as_index=False)
    # calculate frame_{t+1} - frame_{t} and fill NaN which occurs for the first frame of each track
    df['frame_diff'] = gb['frame'].diff().fillna(value=1.0).to_numpy()
    fragmented = df[df['frame_diff'] != 1.0] # df containing all the first frames of fragmentation
    gb_frag = fragmented.groupby('metaId')  # helper for gb.apply
    frag_idx = fragmented.metaId.unique()  # helper for gb.apply
    df['newMetaId'] = df['metaId']  # temporary new metaId

    df = gb.apply(split_at_fragment_lambda, frag_idx, gb_frag)
    df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
    df = df.drop(columns='newMetaId')
    return df


def rot(df, image, k=1):
    '''
    Rotates image and coordinates counter-clockwise by k * 90° within image origin
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :param k: Number of times to rotate by 90°
    :return: Rotated Dataframe and image
    '''
    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
    R = np.array([[c, s], [-s, c]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    for i in range(k):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    return xy, image


def fliplr(df, image):
    '''
    Flip image and coordinates horizontally
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :return: Flipped Dataframe and image
    '''
    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    R = np.array([[-1, 0], [0, 1]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    image = cv2.flip(image, 1)

    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    return xy, image


def augment_data(data, image_path='data/SDD/train', images={}, image_file='reference.jpg', seg_mask=False, use_raw_data=False):
    '''
    Perform data augmentation
    :param data: Pandas df, needs x,y,metaId,sceneId columns
    :param image_path: example - 'data/SDD/val'
    :param images: dict with key being sceneId, value being PIL image
    :param image_file: str, image file name
    :param seg_mask: whether it's a segmentation mask or an image file
    :return:
    '''
    ks = [1, 2, 3]
    for scene in data.sceneId.unique():
        if use_raw_data:
            scene_name, scene_idx = scene.split("_")
            im_path = os.path.join(
                image_path, scene_name, f"video{scene_idx}", image_file)
        else:
            im_path = os.path.join(image_path, scene, image_file)
        if seg_mask:
            im = cv2.imread(im_path, 0)
        else:
            im = cv2.imread(im_path)
        images[scene] = im
    # data without rotation, used so rotated data can be appended to original df
    data_ = data.copy()
    k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
    for k in ks:
        metaId_max = data['metaId'].max()
        for scene in data_.sceneId.unique():
            if use_raw_data:
                im_path = os.path.join(
                    image_path, scene_name, f"video{scene_idx}", image_file)
            else:
                im_path = os.path.join(image_path, scene, image_file)
            if seg_mask:
                im = cv2.imread(im_path, 0)
            else:
                im = cv2.imread(im_path)

            data_rot, im = rot(data_[data_.sceneId == scene], im, k)
            # image
            rot_angle = k2rot[k]
            images[scene + rot_angle] = im

            data_rot['sceneId'] = scene + rot_angle
            data_rot['metaId'] = data_rot['metaId'] + metaId_max + 1
            data = pd.concat([data, data_rot], axis=0)

    metaId_max = data['metaId'].max()
    for scene in data.sceneId.unique():
        im = images[scene]
        data_flip, im_flip = fliplr(data[data.sceneId == scene], im)
        data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
        data_flip['metaId'] = data_flip['metaId'] + metaId_max + 1
        data = pd.concat([data, data_flip], axis=0)
        images[scene + '_fliplr'] = im_flip

    return data, images


def resize_and_pad_image(images, size, pad=2019):
    """ Resize image to desired size and pad image to make it square shaped and ratio of images still is the same, as
    images all have different sizes.
    """
    for key, im in images.items():
        H, W, C = im.shape
        im = cv2.copyMakeBorder(
            im, 0, pad - H, 0, pad - W, cv2.BORDER_CONSTANT)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        images[key] = im


def create_images_dict(unique_scene, image_path, image_file='reference.jpg', use_raw_data=False):
    images = {}
    for scene in unique_scene:
        if image_file == 'oracle.png':
            im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
        else:
            if use_raw_data:
                scene_name, scene_idx = scene.split("_")
                im_path = os.path.join(
                    image_path, scene_name, f"video{scene_idx}", image_file)
            else:
                im_path = os.path.join(image_path, scene, image_file)
            im = cv2.imread(im_path)
        images[scene] = im
    # images channel: blue, green, red 
    return images


def load_images(scenes, image_path, image_file='reference.jpg'):
    images = {}
    if type(scenes) is list:
        scenes = set(scenes)
    for scene in scenes:
        if image_file == 'oracle.png':
            im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
        else:
            im = cv2.imread(os.path.join(image_path, scene, image_file))
        images[scene] = im
    return images


def get_varf_table(df, varf_list, obs_len):
    if obs_len:
        print(f'Computing variation fatcor by obs_len')
    else:
        print(f'Computing variation fatcor by obs_len + pred_len')

    df_varfs = df.groupby(['metaId', 'label', 'sceneId']).size().reset_index()[['metaId', 'label', 'sceneId']]
    df_varfs['scene'] = df_varfs.sceneId.apply(lambda x: x.split('_')[0])
    for varf in varf_list:
        df_stats = aggregate_per_varf_value(df, varf, obs_len)
        df_varfs = df_varfs.merge(df_stats[['metaId', varf]], on='metaId')
    return df_varfs
    

def aggregate_per_varf_value(df, varf, obs_len):
    out = df.groupby('metaId').apply(aggregate_per_varf_value_per_metaId, varf, obs_len)
    df_stats = pd.DataFrame(
        [[idx, item[0], item[1]] for idx, item in out.items()], 
        columns=['metaId', varf, 'label'])
    return df_stats


def aggregate_per_varf_value_per_metaId(df_meta, varf, obs_len):
    x = df_meta["x"].values
    y = df_meta["y"].values

    # sanity check
    unique_labels = np.unique(df_meta["label"].values)
    assert len(unique_labels) == 1
    label = unique_labels[0]

    unique_frame_step = (
        df_meta['frame'].shift(periods=-1) - df_meta['frame']).iloc[:-1].unique()
    assert len(unique_frame_step) == 1
    frame_step = unique_frame_step[0]

    op, attr = varf.split('_')

    # take the observed trajectory, or obs + pred
    if not obs_len:
        obs_len = len(x)
    
    # compute stats
    if attr == 'vel':
        stats_seqs = np.sqrt((x[: obs_len-1] - x[1: obs_len]) ** 2 + \
                             (y[: obs_len-1] - y[1: obs_len]) ** 2) / frame_step
    elif attr == 'acc':
        vel = np.sqrt((x[:obs_len-1] - x[1: obs_len]) ** 2 + \
                      (y[:obs_len-1] - y[1: obs_len]) ** 2) / frame_step
        stats_seqs = (vel[:obs_len-2] - vel[1: obs_len-1]) / frame_step
    elif attr == 'dist':
        stats_seqs = df_meta[:obs_len].dist.apply(
            lambda x: x.min() if not isinstance(x, float) else np.inf)
    elif 'den' in attr:
        stats_seqs = df_meta[:obs_len].dist.apply(
            lambda x: x[x < int(attr[3:])].shape[0] if not isinstance(x, float) else 0)
    else:
        raise ValueError(f'Cannot compute {attr} statistic')

    # take statistic for one sequence
    if op == 'max':
        stats = np.max(stats_seqs)
    elif op == 'avg':
        stats = np.mean(stats_seqs)
    elif op == 'min':
        stats = np.min(stats_seqs)
    elif op == 'abs+max':
        stats = np.max(np.abs(stats_seqs))
    elif op == 'abs+avg':
        stats = np.mean(np.abs(stats_seqs))
    elif op == 'abs+min':
        stats = np.mean(np.abs(stats_seqs))
    elif op == 'tot':
        stats = np.sum(stats_seqs)
    else:
        raise ValueError(f'Cannot compute {op} operation')

    return stats, label


def add_range_column(df, varf, varf_ranges, obs_len, inclusive='both'):
    df_stats = aggregate_per_varf_value(df, varf, obs_len)
    for r in varf_ranges:
        df_stats.loc[df_stats[varf].between(r[0], r[1], inclusive=inclusive), f'{varf}_range'] = f'{r[0]}_{r[1]}'
    df = df.merge(df_stats[['metaId', f'{varf}_range']], on='metaId')
    return df


def convert_df_to_dict(df_gb):
    varf_group_dict = dict()
    for g_range in df_gb.groups.keys():
        df_g = df_gb.get_group(g_range)[['metaId', 'sceneId', 'label']].drop_duplicates()
        assert df_g.metaId.nunique() == df_g.shape[0]
        varf_group_dict[g_range] = df_g.to_dict('list')
    return varf_group_dict


def create_dataset_by_agent_type(
    df, labels, out_dir, statistic_only, 
    same_group_size=False, selected_scenes=None):

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_label = df[df.label.isin(labels)]
    df_gb = df_label.groupby(by='label', dropna=True)
    n_total = df_label[df_label.metaId == df_label.metaId.unique()[0]].shape[0]
    print('Statistics:\n', df_gb.count()['metaId'] / n_total)
    print('# total:', (df_gb.count()['metaId'] / n_total).sum())
    if not statistic_only:
        agent_group_dict = convert_df_to_dict(df_gb)
        for agent, agent_group in agent_group_dict.items():
            if same_group_size:
                min_n = min([len(g["metaId"]) for g in agent_group_dict.values()])
                meta_id_mask = reduce_group_size(agent_group, agent, min_n)
                df_varf = df_label[df_label.metaId.isin(agent_group['metaId'][meta_id_mask])]
            else:
                df_varf = df_label[df_label.metaId.isin(agent_group['metaId'])]
            if selected_scenes is None:
                out_path = os.path.join(out_dir, f"{agent}.pkl")
                df_varf.to_pickle(out_path)
            else:
                df_scenes = pd.DataFrame()
                for scene_id in selected_scenes:
                    out_dir_scene = os.path.join(out_dir, scene_id)
                    out_path = os.path.join(out_dir_scene, f'{agent}.pkl')
                    pathlib.Path(out_dir_scene).mkdir(parents=True, exist_ok=True)

                    df_scene = df_varf[df_varf.sceneId == scene_id]
                    df_scenes = pd.concat([df_scenes, df_scene], axis=0)
                    print(f'scene_id = {scene_id}, label = {agent}, #= {df_scene.metaId.unique().shape[0]}')
                    df_scene.to_pickle(out_path)
                out_dir_scene = os.path.join(out_dir, '__'.join(selected_scenes))
                pathlib.Path(out_dir_scene).mkdir(parents=True, exist_ok=True)
                print(f'scene_id = {selected_scenes}, label = {agent}, #= {df_scenes.metaId.unique().shape[0]}')
                df_scenes.to_pickle(os.path.join(out_dir_scene, f'{agent}.pkl'))
    

def create_dataset_given_range(df, varf, varf_ranges, labels, out_dir, 
        obs_len, statistic_only, inclusive='both', same_group_size=False):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        varf (str | list of str): _description_
        varf_ranges (list of tuple | list of list of tuple): _description_
        labels (_type_): _description_
        out_dir (_type_): _description_
        obs_len (_type_): _description_
        statistic_only (bool): 
            Whether store the generated dataset or give statistic of categorized dataset only.
        inclusive (str):
            Choices = [both, right, left, neither]
        same_group_size (bool, optional): 
            _description_. Defaults to False.
    """
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_label = df[df.label.isin(labels)]

    # categorize by factor of variation
    if isinstance(varf_ranges[0], tuple):
        varf = varf[0]
        df_label = add_range_column(df_label, varf, varf_ranges, obs_len, inclusive=inclusive)
        varf_col_name = f'{varf}_range'
    elif isinstance(varf_ranges[0], list):
        for f, r in zip(varf, varf_ranges):
            df_label = add_range_column(df_label, f, r, obs_len, inclusive=inclusive)
        varf_col_name = '__'.join(varf)+'_range'
        nonan_mask = df_label.isna().any(axis=1)
        df_label.loc[~nonan_mask, varf_col_name] = \
            df_label.loc[~nonan_mask, [f + '_range' for f in varf]].agg('__'.join, axis=1)
    else:
        raise ValueError(f'Cannot process {varf}.')
    df_gb = df_label.groupby(by=varf_col_name, dropna=True)
    print('Statistics:\n', df_gb.count()['metaId'].unique().shape[0])
    print('# total:', (df_gb.count()['metaId'].unique().shape[0]).sum())

    # save categorized dataset
    if not statistic_only:
        varf_group_dict = convert_df_to_dict(df_gb)
        for varf_range, varf_group in varf_group_dict.items():
            if same_group_size:
                min_n = min([len(g["metaId"]) for g in varf_group_dict.values()])
                meta_id_mask = reduce_group_size(varf_group, varf_range, min_n)
                df_varf = df_label[df_label.metaId.isin(varf_group['metaId'][meta_id_mask])]
            else:
                df_varf = df_label[df_label.metaId.isin(varf_group['metaId'])] 
            out_path = os.path.join(out_dir, f"{varf_range}.pkl")
            df_varf.to_pickle(out_path)


def reduce_group_size(varf_group, varf_range, min_n):
    print(f"Group {varf_range}")
    scene_ids, scene_counts = np.unique(
        varf_group["sceneId"], return_counts=True)
    sorted_unique_scene_counts = np.unique(np.sort(scene_counts))
    total_count = 0
    prev_count = 0
    mask = np.zeros_like(scene_counts).astype(bool)
    for scene_count in sorted_unique_scene_counts:
        total_count += (scene_counts >= scene_count).sum() * \
            (scene_count - prev_count)
        if total_count >= min_n:
            break
        mask[scene_counts == scene_count] = True
        prev_count = scene_count
    total_counts = np.zeros_like(scene_counts)
    total_counts[mask] = scene_counts[mask]
    total_counts[mask == False] = prev_count
    less = True
    while less:
        for i in np.where(mask == False)[0]:
            total_counts[i] += min(1, min_n - total_counts.sum())
            if min_n == total_counts.sum():
                less = False
                break
    varf_group["sceneId"] = np.array(varf_group["sceneId"])
    varf_group["metaId"] = np.array(varf_group["metaId"])
    varf_group["label"] = np.array(varf_group["label"])
    meta_id_mask = np.zeros_like(varf_group["metaId"]).astype(bool)
    for scene_idx, scene_id in enumerate(scene_ids):
        scene_count = total_counts[scene_idx]
        scene_mask = varf_group["sceneId"] == scene_id
        scene_labels = varf_group["label"][scene_mask]
        unique_scene_labels, scene_labels_count = np.unique(
            scene_labels, return_counts=True)
        scene_labels_chosen = []
        while len(scene_labels_chosen) < scene_count:
            for label_idx, (unique_scene_label, scene_label_count) in enumerate(zip(unique_scene_labels, scene_labels_count)):
                if scene_label_count != 0:
                    scene_labels_chosen.append(unique_scene_label)
                    scene_labels_count[label_idx] -= 1
                    if len(scene_labels_chosen) == scene_count:
                        break
        labels_chosen, labels_chosen_count = np.unique(
            scene_labels_chosen, return_counts=True)
        for label, label_count in zip(labels_chosen, labels_chosen_count):
            meta_id_idx = np.where(np.logical_and(
                varf_group["label"] == label, varf_group["sceneId"] == scene_id))[0][:label_count]
            meta_id_mask[meta_id_idx] = True
    return meta_id_mask


def compute_distance_with_neighbors(df_scene):
    return df_scene.apply(lambda_distance_with_neighbors, axis=1, df_scene=df_scene)


def lambda_distance_with_neighbors(row, df_scene, step=12):
    # start = datetime.datetime.now()
    frame_diff = df_scene.frame - row.frame
    df_sim = df_scene[(frame_diff < step/2) & \
        (frame_diff >= -step/2) & (df_scene.metaId != row.metaId)]
    dist = np.inf if df_sim.shape[0] == 0 else compute_distance_xy(df_sim, row.x, row.y)
    # duration = datetime.datetime.now() - start
    # print(f'### meta_id = {row.metaId}, time = {duration}')
    return dist


def compute_distance_xy(df_sim, x, y):
    x_diff = df_sim['x'] - x
    y_diff = df_sim['y'] - y
    dist = np.sqrt((x_diff ** 2 + y_diff ** 2))
    return np.array(dist)


def plot_varf_histograms(df_varf, out_dir):
    stats_all = np.array([])
    varf = df_varf.columns[-1]
    # Visualize data
    for label, indices in df_varf.groupby('label').groups.items():
        if label not in ["Pedestrian", "Biker"]:
            continue
        stats = df_varf.iloc[indices].loc[:, varf].values
        plot_histogram(stats, f'{label}_{varf}', out_dir)
        stats_all = np.append(stats_all, stats)
    plot_histogram(stats_all, f"Mixed_{varf}", out_dir)


def plot_varf_hist_obs_and_complete(df_varf, out_dir):
    varf_obs, varf_com = df_varf.columns[-2], df_varf.columns[-1]
    data_all_diff, data_all_obs, data_all_com = np.array([]), np.array([]), np.array([])
    # Visualize data
    for label, indices in df_varf.groupby('label').groups.items():
        if label not in ["Pedestrian", "Biker"]:
            continue
        data_obs = df_varf.iloc[indices].loc[:, varf_obs].values
        data_com = df_varf.iloc[indices].loc[:, varf_com].values
        data_diff = data_obs - data_com
        plot_histogram(data_diff, f'{label}_{varf_obs}_element_diff', out_dir)
        plot_histogram_overlay(data_obs, data_com, f'{label}_{varf_obs}_distr_diff', out_dir)
        data_all_diff = np.append(data_all_diff, data_diff)
        data_all_obs = np.append(data_all_obs, data_obs)
        data_all_com = np.append(data_all_com, data_com)
    plot_histogram(data_all_diff, f"Mixed_{varf_obs}_element_diff", out_dir)
    plot_histogram_overlay(data_all_obs, data_all_com, f'Mixed_{varf_obs}_distr_diff', out_dir)


def plot_histogram(data, title, out_dir, format='png'):
    fig = plt.figure()
    data, stats = filter_long_tail_arr(data)
    mean, std, min, max, p_zero, p_filter = stats
    sns.histplot(data, kde=True)
    plt.title(
        f"{title}, \nMean: {mean}, Std: {std}, Min: {min}, Max: {max}, Zero: {p_zero}, Filter: {p_filter}")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title+'.'+format))
    plt.close(fig)


def plot_histogram_overlay(data_obs, data_com, title, out_dir, format='png'):
    fig = plt.figure()
    data_obs, _ = filter_long_tail_arr(data_obs)
    data_com, _ = filter_long_tail_arr(data_com)
    data_obs = data_obs[data_obs != 0]
    data_com = data_com[data_com != 0]
    df_obs = pd.DataFrame(data_obs, columns=['value'])
    df_obs['type'] = 'observation'
    df_com = pd.DataFrame(data_com, columns=['value'])
    df_com['type'] = 'complete'
    df_cat = pd.concat([df_obs, df_com], axis=0)
    df_cat = df_cat.reset_index(drop=True)
    sns.histplot(data=df_cat, x='value', hue="type")
    plt.title(title)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title+'.'+format))
    plt.close(fig)


def plot_pairplot(df_varfs, varf_list, label, title, out_dir, kind='kde', format='png'):
    if label == 'Mixed':
        df_label = df_varfs[
            (df_varfs.label == 'Pedestrian') | (df_varfs.label == 'Biker')]
    elif label == 'All':
        df_label = df_varfs
    else:
        df_label = df_varfs[df_varfs.label == label]

    fig = plt.figure()
    df_label_filter, p_filter = filter_long_tail_df(
        df_label[['metaId', 'scene', 'label']+varf_list], varf_list)
    plot_kws = dict(s=1) if kind == 'scatter' else None
    sns.pairplot(
        data=df_label_filter, 
        hue="scene", 
        vars=varf_list, 
        plot_kws=plot_kws,
        diag_kind="hist",
        kind=kind
    )
    title = f'{title}_{label}_{kind}_{str(p_filter)}'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title+'.'+format))
    plt.close(fig)


def plot_jointplot(df_varfs, varf_list, label, title, out_dir, hue, kind='kde', format='png'):
    if label == 'Mixed':
        df_label = df_varfs[
            (df_varfs.label == 'Pedestrian') | (df_varfs.label == 'Biker')]
    elif label == 'All':
        df_label = df_varfs
    else:
        df_label = df_varfs[df_varfs.label == label]

    for i, varf1 in enumerate(varf_list):
        for j, varf2 in enumerate(varf_list):
            if i < j:
                fig = plt.figure()
                df_label_filter, p_filter = filter_long_tail_df(
                    df_label[['metaId', 'scene', 'label', varf1, varf2]], [varf1, varf2])
                try:
                    sns.jointplot(data=df_label_filter, x=varf1, y=varf2, 
                        kind=kind, hue=hue)
                except np.linalg.LinAlgError:
                    kind = 'scatter'
                    sns.jointplot(data=df_label_filter, x=varf1, y=varf2, 
                        kind=kind, hue=hue)
                except:
                    print('Error!')
                title_save = f'{title}_{hue}_{label}_{varf1}_{varf2}_{kind}_{str(p_filter)}.{format}'
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(os.path.join(out_dir, title_save))
                plt.close(fig)


def plot_scene_w_numeric(df_varfs, varf, title, out_dir, format='png'):
    df_filter, p_filter = filter_long_tail_df(
        df_varfs[['metaId', 'scene', 'label', varf]], [varf])
    # filter out scene "quad"
    df_filter = df_filter[df_filter.scene != 'quad']

    unique_scenes = df_filter.scene.unique()
    n_scene = unique_scenes.shape[0]
    fig, axs = plt.subplots(4, n_scene+1, 
        figsize=(4*(n_scene+1), 16), sharex=True, sharey=True)
    binwidth = df_filter[varf].max() / 30
    for c, scene in enumerate(unique_scenes):
        data = df_filter[(df_filter.scene == scene)]
        axs[0, c].set_title(unique_scenes[c])
        # pedestrain
        sns.histplot(data=data[data.label == 'Pedestrian'], x=varf, 
            ax=axs[0, c], stat='probability', binwidth=binwidth)
        # biker 
        sns.histplot(data=data[data.label == 'Biker'], x=varf, 
            ax=axs[1, c], stat='probability', binwidth=binwidth)
        # mixed 
        sns.histplot(data=data[(data.label == 'Pedestrian') | (data.label == 'Biker')], x=varf, 
            ax=axs[2, c], hue='label', stat='probability', 
            hue_order=['Biker', 'Pedestrian'], binwidth=binwidth)
        # all 
        sns.histplot(data=data, x=varf, 
            ax=axs[3, c], stat='probability', binwidth=binwidth)
    axs[0, -1].set_title('All scenes')
    # pedestrain
    sns.histplot(data=df_filter[df_filter.label == 'Pedestrian'], x=varf, 
        ax=axs[0, -1], stat='probability', binwidth=binwidth)
    # biker 
    sns.histplot(data=df_filter[df_filter.label == 'Biker'], x=varf, 
        ax=axs[1, -1], stat='probability', binwidth=binwidth)
    # mixed 
    sns.histplot(data=df_filter[(df_filter.label == 'Pedestrian') | (df_filter.label == 'Biker')], x=varf, 
        ax=axs[2, -1], hue='label', stat='probability', 
        hue_order=['Biker', 'Pedestrian'], binwidth=binwidth)
    # all 
    sns.histplot(data=df_filter, x=varf, 
        ax=axs[3, -1], stat='probability', binwidth=binwidth)
    axs[0, 0].set_ylabel('Pedestrian')
    axs[1, 0].set_ylabel('Biker')
    axs[2, 0].set_ylabel('Pedestrian + Biker')
    axs[3, 0].set_ylabel('All agent types')
    plt.tight_layout()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{title}_scene_w_{varf}_{p_filter}_noquad.{format}'))
    plt.close(fig)


def filter_long_tail_arr(arr, n=3):
    # for statistics computing
    n_data = arr.shape[0]
    arr = arr[~np.isnan(arr) & (arr != np.inf)]
    if arr.shape[0]:
        mean = np.round(np.mean(arr), 2)
        std = np.round(np.std(arr), 2)
        min = np.round(np.min(arr), 2)
        max = np.round(np.max(arr), 2)
    else:
        raise ValueError('stats array is empty')
    p_zero = np.round((arr == 0).sum() / n_data, 2)
    arr = arr[
        (arr < mean + n * std) & (arr > mean - n * std) & (arr != 0)]
    p_filter = np.round((n_data - arr.shape[0]) / n_data, 2)
    return arr, (mean, std, min, max, p_zero, p_filter)


def filter_long_tail_series(series, n=3):
    full_index = series.index
    series = series[~series.isnull() & (series != np.inf)]
    if series.shape[0]:
        mean = np.round(series.mean(), 2)
        std = np.round(series.std(), 2)
    else:
        raise ValueError('Series is empty')
    series = series[
        (series < mean + n * std) & (series > mean - n * std) & (series != 0)]
    return full_index.drop(series.index)


def filter_long_tail_df(df_varfs, varf_list, n=3):
    idx_out = pd.Index([])
    for varf in varf_list:
        idx_out = idx_out.append(filter_long_tail_series(df_varfs[varf]))
    idx_out_unique = idx_out.unique()
    df_varfs_filter = df_varfs.drop(idx_out_unique)
    p_filter = round(len(idx_out_unique) / df_varfs.shape[0], 2)
    return df_varfs_filter, p_filter


def split_train_val_test_sequentially(
    data_path, train_files, val_split, test_splits=None, 
    shuffle=False, share_val_test=False):
    print(f"Split {train_files} given val_split={val_split}, test_split={test_splits}")
    df_train, df_val, df_test = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
    for train_file, test_split in zip(train_files, test_splits):
        df_train_ = pd.read_pickle(os.path.join(data_path, train_file))
        df_train_, df_val_, df_test_ = dataset_split_by_ratio(
            df_train_, val_split, test_split, shuffle, share_val_test)
        df_train = pd.concat([df_train, df_train_])
        df_val = pd.concat([df_val, df_val_])
        if df_test_ is not None:
            df_test = pd.concat([df_test, df_test_])
    return df_train, df_val, df_test


def dataset_split_by_ratio(df, val_split, test_split=None, 
    shuffle=False, share_val_test=False, given_test_meta_ids=None):
    # idx
    unique_meta_ids = np.unique(df["metaId"])
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(unique_meta_ids)
    n_metaId = unique_meta_ids.shape[0]
    n_val = int(val_split) if val_split > 1 else int(val_split * n_metaId)
    # split
    if test_split is not None:
        n_test = int(test_split) if test_split > 1 else int(test_split * n_metaId)
        if share_val_test:
            print('Share validation and test set')
            n_train = n_metaId - n_test 
            train_meta_ids, test_meta_ids = np.split(unique_meta_ids, [n_train])
            if n_val != 0:
                interval = n_test // n_val if n_test // n_val > 1 else 3
                val_meta_ids = test_meta_ids[::interval]
                df_val = reduce_df_meta_ids(df, val_meta_ids)
            else:
                df_val = None
            df_test = reduce_df_meta_ids(df, test_meta_ids)
        else:
            print('Validation and test sets are independent')
            n_train = n_metaId - n_val - n_test 
            train_meta_ids, val_meta_ids, test_meta_ids = np.split(unique_meta_ids, [n_train, n_train + n_val])
            if given_test_meta_ids is not None:
                test_meta_ids = given_test_meta_ids
                print('Replaced test set by given test meta ids')
            df_test = reduce_df_meta_ids(df, test_meta_ids)
            df_val = reduce_df_meta_ids(df, val_meta_ids)
    else:
        n_train = n_metaId - n_val
        val_meta_ids, train_meta_ids = np.split(
            unique_meta_ids, [n_train])
        df_test = None
        df_val = reduce_df_meta_ids(df, val_meta_ids)
    df_train = reduce_df_meta_ids(df, train_meta_ids)
    return df_train, df_val, df_test


def reduce_df_meta_ids(df, meta_ids):
    return df[(df["metaId"].values == meta_ids[:, None]).sum(axis=0).astype(bool)]


def dataset_split_given_scenes(data_path, files, scenes):
    print(f"Split {files} given scenes={scenes}")
    df = pd.concat([pd.read_pickle(os.path.join(data_path, file)) for file in files])
    df_selected = df[df.sceneId.isin(scenes)]
    return df_selected


def split_train_val_test_randomly(data_dir, data_filename, val_split, test_split, seed=1):
    """
    Generate train / val / test set randomly. 
    It will output train.pkl / val.pkl / test.pkl under the same directory of input data file. 
    """
    data_folder = data_filename.replace('.pkl', '')
    out_dir = f'{data_dir}/{data_folder}'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_pickle(f'{data_dir}/{data_filename}')
    unique_meta_ids = np.unique(df["metaId"])
    
    n_data = unique_meta_ids.shape[0]
    n_val = int(val_split) if val_split > 1 else int(val_split * n_data)
    n_test = int(test_split) if test_split > 1 else int(test_split * n_data)
    n_train = n_data - n_val - n_test

    set_random_seeds(seed)
    np.random.shuffle(unique_meta_ids)
    
    train_meta_ids, val_meta_ids, test_meta_ids = \
        np.split(unique_meta_ids, [n_train, n_train + n_val])
    df_train = reduce_df_meta_ids(df, train_meta_ids)
    df_val = reduce_df_meta_ids(df, val_meta_ids)
    df_test = reduce_df_meta_ids(df, test_meta_ids)
    print(f'# data = {unique_meta_ids.shape[0]}')
    print(f'# train = {train_meta_ids.shape[0]}')
    print(f'# val = {val_meta_ids.shape[0]}')
    print(f'# test = {test_meta_ids.shape[0]}')

    df_train.to_pickle(f'{out_dir}/train.pkl')
    df_val.to_pickle(f'{out_dir}/val.pkl')
    df_test.to_pickle(f'{out_dir}/test.pkl')
    print('Split train/val/test set')


def load_predefined_train_val_test(data_path, batch_size, n_train_batch=None, shuffle=False):
    df_train = pd.read_pickle(f'{data_path}/train.pkl')
    df_val = pd.read_pickle(f'{data_path}/val.pkl')
    df_test = pd.read_pickle(f'{data_path}/test.pkl')
    if n_train_batch is not None:
        n_sample = int(batch_size * n_train_batch)
        unique_train_ids = df_train.metaId.unique()
        n_train = unique_train_ids.shape[0]
        assert n_sample <= n_train, \
            f'Training set size ({n_train}) < Sample size ({n_sample})'
        if shuffle:
            np.random.shuffle(unique_train_ids)
        df_train = reduce_df_meta_ids(df_train, unique_train_ids[:n_sample])
    return df_train, df_val, df_test 


def prepare_dataeset(
    data_path, load_data, batch_size, n_train_batch, 
    train_files, val_files, val_split, test_splits, 
    shuffle, share_val_test, mode='train', show_details=False):
    if load_data == 'predefined':
        print('Loading predefined train/val/test sets')
        df_train, df_val, df_test = load_predefined_train_val_test(data_path, 
            batch_size=batch_size, n_train_batch=n_train_batch, shuffle=shuffle)
    else:
        print('Splitting train/val/test sets sequentially')
        if mode == 'train':
            assert train_files is not None, 'No train file is provided'
            assert val_files is not None, 'No val file is provided'
            assert val_split is not None, 'No val split is provided'
            if train_files == val_files:
                df_train, df_val, df_test = split_train_val_test_sequentially(data_path, 
                    train_files, val_split, test_splits, shuffle, share_val_test)
            else:
                raise NotImplementedError 
            df_train = limit_samples(df_train, n_train_batch, batch_size)
        elif mode == 'eval':
            assert val_files is not None, 'No val file is provided'
            df_train, df_val, df_test = split_train_val_test_sequentially(data_path, 
                val_files, val_split, test_splits, shuffle, share_val_test)
        else:
            raise NotImplementedError

    if show_details:
        print(f'train_meta_ids: {df_train.metaId.unique()}')
        print(f'val_meta_ids: {df_val.metaId.unique()}')
        print(f'test_meta_ids: {df_test.metaId.unique()}')

    if mode == 'train':
        if df_train is not None: print(f"df_train: {df_train.shape}; #={df_train.metaId.unique().shape[0]}")
        if df_val is not None: print(f"df_val: {df_val.shape}; #={df_val.metaId.unique().shape[0]}")
    if df_test is not None: print(f"df_test: {df_test.shape}; #={df_test.metaId.unique().shape[0]}")

    return df_train, df_val, df_test 

def get_meta_ids_focus(df=None, given_meta_ids=None, given_csv=None, random_n=None):
    if given_meta_ids is not None:
        if isinstance(given_meta_ids, int):
            meta_ids_focus = [given_meta_ids]
        elif isinstance(given_meta_ids, list):
            meta_ids_focus = given_meta_ids
        else:
            raise ValueError(f'Invalid given_meta_ids={given_meta_ids}')
    elif given_csv['path'] is not None:
        path = given_csv['path']
        col1, col2, op = given_csv['name'].split('__')
        n_limited = given_csv['n_limited']
        df_result = pd.read_csv(path)
        if op == 'diff':
            df_result.loc[:, 'diff'] = df_result[col1].values - df_result[col2].values
        elif op == 'abs_diff':
            df_result.loc[:, 'diff'] = np.abs(df_result[col1].values - df_result[col2].values)
        else:
            raise ValueError(f'Invalid op={op}')
        meta_ids_focus = df_result.sort_values(
            by='diff', ascending=False).head(n_limited).metaId.values
    elif random_n is not None:
        unique_meta_ids = df.metaId.unique() 
        np.random.shuffle(unique_meta_ids)
        meta_ids_focus = unique_meta_ids[:random_n]
    else:
        meta_ids_focus = df.metaId.unique() 
    print('Focusing on meta_ids=', meta_ids_focus)
    return meta_ids_focus


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    cv2.setRNGSeed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def limit_samples(df, num, batch_size, random_ids=True):
    if num is None:
        return df
    num_total = num * batch_size
    meta_ids = np.unique(df["metaId"])
    if random_ids:
        np.random.shuffle(meta_ids)
    meta_ids = meta_ids[:num_total]
    df = reduce_df_meta_ids(df, meta_ids)
    return df
