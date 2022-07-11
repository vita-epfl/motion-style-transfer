import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.image_utils import get_patch, sampling, image2world, swap_pavement_terrain
from utils.kmeans import kmeans


def torch_multivariate_gaussian_heatmap(coordinates, H, W, dist, sigma_factor, ratio, device, rot=False):
    """
    Create Gaussian Kernel for CWS
    """
    ax = torch.linspace(0, H, H, device=device) - coordinates[1]
    ay = torch.linspace(0, W, W, device=device) - coordinates[0]
    xx, yy = torch.meshgrid([ax, ay])
    meshgrid = torch.stack([yy, xx], dim=-1)
    radians = torch.atan2(dist[0], dist[1])

    c, s = torch.cos(radians), torch.sin(radians)
    R = torch.Tensor([[c, s], [-s, c]]).to(device)
    if rot:
        R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
    # some small padding to avoid division by zero
    dist_norm = dist.square().sum(-1).sqrt() + 5

    conv = torch.Tensor([[dist_norm / sigma_factor / ratio, 0],
                        [0, dist_norm / sigma_factor]]).to(device)
    conv = torch.square(conv)
    T = torch.matmul(R, conv)
    T = torch.matmul(T, R.T)

    kernel = (torch.matmul(meshgrid, torch.inverse(T)) * meshgrid).sum(-1)
    kernel = torch.exp(-0.5 * kernel)
    return kernel / kernel.sum()


def evaluate(
    model, val_loader, val_images, device, 
    dataset_name, homo_mat, input_template, waypoints, mode, 
    n_goal, n_traj, obs_len, batch_size, resize_factor=0.25,  
    temperature=1, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None, 
    return_preds=False, return_samples=False, network=None, swap_semantic=False):
    """

    :param model: torch model
    :param val_loader: torch dataloader
    :param val_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
    :param n_goal: int, number of goals
    :param n_traj: int, number of trajectories per goal
    :param obs_len: int, observed timesteps
    :param batch_size: int, batch_size
    :param device: torch device
    :param input_template: torch.Tensor, heatmap template
    :param waypoints: number of waypoints
    :param resize_factor: resize factor
    :param temperature: float, temperature to control peakiness of heatmap
    :param use_TTST: bool
    :param use_CWS: bool
    :param rel_thresh: float
    :param CWS_params: dict
    :param dataset_name: ['sdd','ind','eth']
    :param params: dict with hyperparameters
    :param homo_mat: dict with homography matrix
    :param mode: ['val', 'test']
    :return: val_ADE, val_FDE for one epoch
    """

    model.eval()
    counter = 0
    val_ade_list, val_fde_list, meta_id_list, scene_id_list = [], [], [], []
    df_out = pd.DataFrame()

    # variables for visualization
    if return_preds:
        if return_samples:
            trajs_dict = {'groundtruth': [], 'prediction': [], 'waypoint_sample': [], 'goal_map': [], 'goal_sigmoid_map': []} 
        else:
            trajs_dict = {'groundtruth': [], 'prediction': []} 
    else:
        trajs_dict = None 

    with torch.no_grad():
        # outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
        for trajectory, df_batch, scene_id in val_loader:
            # Get scene image and apply semantic segmentation
            scene_image = val_images[scene_id].to(device).unsqueeze(0)
            scene_image = model.segmentation(scene_image)
            
            # possibly adapt semantic image 
            scene_image = model.adapt_semantic(scene_image)
            meta_ids = df_batch[0].metaId.unique()
            n_data = trajectory.shape[0]

            # swap semantic layers if needed
            if swap_semantic:
                scene_image = swap_pavement_terrain(scene_image)

            # add scene embedding
            if network == 'embed':
                scene_image = model.scene_embedding(scene_image)

            if dataset_name == 'eth': 
                print(counter)
                counter += batch_size
				# Break after certain number of batches to approximate evaluation, else one epoch takes really long
                if counter > 30 and mode == 'val':
                    break

            for b in range(0, len(trajectory), batch_size):
                # Create Heatmaps for past and ground-truth future trajectories
                _, _, H, W = scene_image.shape
                observed = trajectory[b:b+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()  # (batch_size*obs_len, n_coord)
                observed_map = get_patch(input_template, observed, H, W)  # batch_size*obs_len list of (height, width)
                observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])  # (batch_size, obs_len, height, width)

                gt_future = trajectory[b:b+batch_size, obs_len:].to(device)  # (batch_size, pred_len)
                semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)  # (batch_size, n_class, height, width)

                # add embedding 
                if network == 'embed':
                    observed_map = model.motion_embedding(observed_map)

                # Forward pass
                features = model.pred_features(semantic_image, observed_map)  # n_layer=6 list of (batch_size, n_channel, height, width)

                # Predict goal and waypoint probability distributions
                pred_goal_map = model.pred_goal(features)  # (batch_size, pred_len, height, width)
                pred_waypoint_map = pred_goal_map[:, waypoints]  # (batch_size, n_waypoints, height, width)

                pred_waypoint_map_sigmoid = pred_waypoint_map / temperature 
                pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)  # (batch_size, n_waypoints, height, width)

                ################################################ TTST ##################################################
                if use_TTST:
                    # TTST Begin
                    # sample a large amount of goals to be clustered
                    goal_samples = sampling(
                        pred_waypoint_map_sigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
                    goal_samples = goal_samples.permute(2, 0, 1, 3)

                    num_clusters = n_goal - 1
                    goal_samples_softargmax = model.softargmax(
                        pred_waypoint_map[:, -1:])  # first sample is softargmax sample

                    # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
                    goal_samples_list = []
                    for person in range(goal_samples.shape[1]):
                        goal_sample = goal_samples[:, person, 0]

                        # Actual k-means clustering, Outputs:
                        # cluster_ids_x -  Information to which cluster_idx each point belongs to
                        # cluster_centers - list of centroids, which are our new goal samples
                        cluster_ids_x, cluster_centers = kmeans(
                            X=goal_sample, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False, tol=0.001, iter_limit=1000)
                        goal_samples_list.append(cluster_centers)

                    goal_samples = torch.stack(
                        goal_samples_list).permute(1, 0, 2).unsqueeze(2)
                    goal_samples = torch.cat(
                        [goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
                    # TTST End

                # Not using TTST
                else:
                    goal_samples = sampling(
                        pred_waypoint_map_sigmoid[:, -1:], num_samples=n_goal)  # (batch_size, 1, n_goal, n_coord)
                    goal_samples = goal_samples.permute(2, 0, 1, 3)  # (n_goal, batch_size, 1, n_coord)

                # Predict waypoints:
                ################################################ CWS ###################################################
                # CWS Begin
                if use_CWS and len(waypoints) > 1:
                    sigma_factor = CWS_params['sigma_factor']
                    ratio = CWS_params['ratio']
                    rot = CWS_params['rot']

                    goal_samples = goal_samples.repeat(n_traj, 1, 1, 1)  # repeat K_a times
                    # [N, 2]
                    last_observed = trajectory[b:b+batch_size, obs_len-1].to(device)
                    # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
                    waypoint_samples_list = []
                    for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
                        waypoint_list = []  # for each K sample have a separate list
                        waypoint_list.append(waypoint_samples)

                        for waypoint_num in reversed(range(len(waypoints)-1)):
                            distance = last_observed - waypoint_samples
                            gaussian_heatmaps = []
                            traj_idx = g_num // n_goal  # idx of trajectory for the same goal
                            # for each person
                            for dist, coordinate in zip(distance, waypoint_samples):
                                length_ratio = 1 / (waypoint_num + 2)
                                # Get the intermediate point's location using CV model
                                gauss_mean = coordinate + (dist * length_ratio)
                                sigma_factor_ = sigma_factor - traj_idx
                                gaussian_heatmaps.append(torch_multivariate_gaussian_heatmap(
                                    gauss_mean, H, W, dist, sigma_factor_, ratio, device, rot))
                            gaussian_heatmaps = torch.stack(
                                gaussian_heatmaps)  # [N, H, W]

                            waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
                            waypoint_map = waypoint_map_before * gaussian_heatmaps
                            # normalize waypoint map
                            waypoint_map = (waypoint_map.flatten(
                                1) / waypoint_map.flatten(1).sum(-1, keepdim=True)).view_as(waypoint_map)

                            # For first traj samples use softargmax
                            if g_num // n_goal == 0:
                                # Softargmax
                                waypoint_samples = model.softargmax_on_softmax_map(
                                    waypoint_map.unsqueeze(0))
                                waypoint_samples = waypoint_samples.squeeze(0)
                            else:
                                waypoint_samples = sampling(waypoint_map.unsqueeze(
                                    1), num_samples=1, rel_threshold=0.05)
                                waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                                waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
                            waypoint_list.append(waypoint_samples)

                        waypoint_list = waypoint_list[::-1]
                        waypoint_list = torch.stack(waypoint_list).permute(
                            1, 0, 2)  # permute back to [N, # waypoints, 2]
                        waypoint_samples_list.append(waypoint_list)
                    waypoint_samples = torch.stack(waypoint_samples_list)

                    # CWS End

                # If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
                elif not use_CWS and len(waypoints) > 1:
                    waypoint_samples = sampling(
                        pred_waypoint_map_sigmoid[:, :-1], num_samples=n_goal * n_traj)
                    waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                    goal_samples = goal_samples.repeat(n_traj, 1, 1, 1)  # repeat K_a times
                    waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)

                # in case len(waypoints) == 1, only goal is needed (goal counts as one waypoint in this implementation)
                elif len(waypoints) == 1:
                    waypoint_samples = goal_samples

                if return_samples:
                    trajs_dict['goal_map'].append(pred_goal_map.cpu().detach().numpy())
                    trajs_dict['goal_sigmoid_map'].append(model.sigmoid(pred_goal_map / temperature).cpu().detach().numpy())
                    # (n_goal, batch_size, 1, n_coord) -> (batch_size, 1, n_goal, n_coord)
                    trajs_dict['waypoint_sample'].append(
                        waypoint_samples.permute(1, 2, 0, 3).cpu().detach().numpy())  

                # Interpolate trajectories given goal and waypoints
                trajs_samples = []
                for waypoint in waypoint_samples:
                    waypoint_map = get_patch(
                        input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W)  # batch_size list of (height, width)
                    waypoint_map = torch.stack(waypoint_map).reshape(
                        [-1, len(waypoints), H, W])  # (batch_size, n_waypoint, height, width)

                    waypoint_maps_downsampled = [nn.AvgPool2d(
                        kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))]  # n_layer-1=5 list of (batch_size, n_waypoint, height_down, width_down)
                    waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled  # n_layer=6 list of (batch_size, n_waypoint, height_down, width_down)

                    traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(
                        features, waypoint_maps_downsampled)]  # n_layer list of (batch_size, n_channel+n_waypoint, height, width)

                    # predict trajectory
                    pred_traj_map = model.pred_traj(traj_input)  # (batch_size, pred_len, height, width)
                    pred_traj = model.softargmax(pred_traj_map)  # (batch_size, pred_len, n_coord)
                    trajs_samples.append(pred_traj)
                trajs_samples = torch.stack(trajs_samples)  # (n_goal, batch_size, pred_len, n_coord)
                
                gt_goal = gt_future[:, -1:]  # (batch_size, 1, n_coord)

                # converts ETH/UCY pixel coordinates back into world-coordinates
                if dataset_name == 'eth':
                    waypoint_samples = image2world(waypoint_samples, scene_id, homo_mat, resize_factor)
                    pred_traj = image2world(pred_traj, scene_id, homo_mat, resize_factor)
                    gt_future = image2world(gt_future, scene_id, homo_mat, resize_factor)
                
                ade_batch = ((((gt_future - trajs_samples) / resize_factor) ** 2).sum(dim=3) ** 0.5).mean(dim=2)
                fde_batch = ((((gt_goal - waypoint_samples[:, :, -1:]) / resize_factor) ** 2).sum(dim=3) ** 0.5)
                
                if return_preds:
                    if b == 0:
                        trajs_dict['groundtruth'].append(
                            trajectory.cpu().detach().numpy() / resize_factor)  # (batch_size, tot_len, n_coor)
                    # take the most accurate prediction only 
                    best_indices = ade_batch.argmin(axis=0)
                    trajs_dict['prediction'].append((trajs_samples[
                        best_indices, range(trajs_samples.shape[1]), ...] 
                        / resize_factor).cpu().detach().numpy())  # (n_goal, batch_size, n_coor)

                # TODO: a potential bug? ADE and FDE may not take the same traj prediction 
                ade = ade_batch.min(dim=0)[0].cpu().detach().numpy()  # (batch_size, )
                fde = fde_batch.min(dim=0)[0][:,0].cpu().detach().numpy()  # (batch_size, )
                val_ade_list.append(ade)
                val_fde_list.append(fde)
            meta_id_list.append(meta_ids)
            scene_id_list.append([scene_id]*n_data)

        val_ade_arr = np.concatenate(val_ade_list)
        val_fde_arr = np.concatenate(val_fde_list)
        val_ade_avg = val_ade_arr.mean()
        val_fde_avg = val_fde_arr.mean()
        
        meta_id_ready = np.concatenate(meta_id_list)
        scene_id_ready = sum(scene_id_list, [])
        df_out.loc[:, 'metaId'] = meta_id_ready
        df_out.loc[:, 'sceneId'] = scene_id_ready
        df_out.loc[:, 'ade'] = val_ade_arr
        df_out.loc[:, 'fde'] = val_fde_arr

    if return_preds:
        for key, value in trajs_dict.items():
            trajs_dict[key] = np.concatenate(value, axis=0)
        trajs_dict['metaId'] = meta_id_ready
        trajs_dict['sceneId'] = scene_id_ready

    return val_ade_avg, val_fde_avg, df_out, trajs_dict
