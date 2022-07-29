import torch
import torch.nn as nn
import numpy as np 
from utils.image_utils import get_patch, swap_pavement_terrain
from utils.data_utils import set_random_seeds 


def train_epoch(
    model, train_loader, train_images, optimizer, criterion, loss_scale, device, 
    dataset_name, homo_mat, gt_template, input_template, waypoints, 
    epoch, obs_len, pred_len, batch_size, e_unfreeze, resize_factor, 
    network=None, swap_semantic=False):
    """
    Run training for one epoch

    :param model: torch model
    :param train_loader: torch dataloader
    :param train_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
    :param e: epoch number
    :param params: dict of hyperparameters
    :param gt_template:  precalculated Gaussian heatmap template as torch.Tensor
    :return: train_ADE, train_FDE, train_loss for one epoch
    """
    train_loss = 0
    train_ADE = []
    train_FDE = []
    model.train()
    counter = 0

    # outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
    for batch, (trajectory, meta, scene) in enumerate(train_loader):
        
        # Get scene image and apply semantic segmentation
        if epoch < e_unfreeze:  # before unfreeze only need to do semantic segmentation once
            model.eval()
            scene_image = train_images[scene].to(device).unsqueeze(0)
            scene_image = model.segmentation(scene_image)
            model.train()

        counter += len(trajectory)
        # print('Batch number of trajectories: {:d}'.format(len(trajectory)))

        # inner loop, for each trajectory in the scene
        for i in range(0, len(trajectory), batch_size):
            # set_random_seeds(1)
            # np.save(f'z_run1/{scene}__batch_{i}__epoch_{epoch}.npy', 
            #     trajectory[i:i+batch_size, :obs_len, :].detach().cpu().numpy())
            
            if epoch >= e_unfreeze:
                scene_image = train_images[scene].to(device).unsqueeze(0)
                scene_image = model.segmentation(scene_image)

            # possibly adapt semantic image 
            semantic_img = model.adapt_semantic(scene_image)

            # swap semantic layers if needed
            if swap_semantic:
                semantic_img = swap_pavement_terrain(semantic_img)

            # Create Heatmaps for past and ground-truth future trajectories
            _, _, H, W = scene_image.shape  # image shape

            observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
            observed_map = get_patch(input_template, observed, H, W)
            observed_map = torch.stack(
                observed_map).reshape([-1, obs_len, H, W])

            gt_future = trajectory[i:i + batch_size, obs_len:].to(device)
            gt_future_map = get_patch(
                gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
            gt_future_map = torch.stack(
                gt_future_map).reshape([-1, pred_len, H, W])

            gt_waypoints = gt_future[:, waypoints]
            gt_waypoint_map = get_patch(
                input_template, gt_waypoints.reshape(-1, 2).cpu().numpy(), H, W)
            gt_waypoint_map = torch.stack(gt_waypoint_map).reshape(
                [-1, gt_waypoints.shape[1], H, W])
            
            # add embedding 
            if network == 'embed':
                semantic_img = model.scene_embedding(semantic_img)
                observed_map = model.motion_embedding(observed_map)

            # Concatenate heatmap and semantic map
            semantic_map = semantic_img.expand(
                observed_map.shape[0], -1, -1, -1)  # expand to match heatmap size
            
            # forward pass 
            features = model.pred_features(semantic_map, observed_map)

            # Predict goal and waypoint probability distribution
            pred_goal_map = model.pred_goal(features)
            goal_loss = criterion(pred_goal_map, gt_future_map) * loss_scale  # BCEWithLogitsLoss

            # Prepare (downsample) ground-truth goal and trajectory heatmap representation for conditioning trajectory decoder
            gt_waypoints_maps_downsampled = [nn.AvgPool2d(
                kernel_size=2**i, stride=2**i)(gt_waypoint_map) for i in range(1, len(features))]
            gt_waypoints_maps_downsampled = [
                gt_waypoint_map] + gt_waypoints_maps_downsampled

            # Predict trajectory distribution conditioned on goal and waypoints
            traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(
                features, gt_waypoints_maps_downsampled)]
            pred_traj_map = model.pred_traj(traj_input)
            traj_loss = criterion(pred_traj_map, gt_future_map) * loss_scale  # BCEWithLogitsLoss

            # Backprop
            loss = goal_loss + traj_loss
            # torch.set_printoptions(precision=16)
            # print(goal_loss.detach().cpu(), traj_loss.detach().cpu(), loss.detach().cpu(), 
            #     goal_loss.detach().cpu()+traj_loss.detach().cpu()==loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss
                # Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
                pred_traj = model.softargmax(pred_traj_map)
                pred_goal = model.softargmax(pred_goal_map[:, -1:])

                train_ADE.append(
                    ((((gt_future - pred_traj) / resize_factor) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
                train_FDE.append(
                    ((((gt_future[:, -1:] - pred_goal[:, -1:]) / resize_factor) ** 2).sum(dim=2) ** 0.5).mean(dim=1))

    # print('train ADE:', [x.detach().cpu().numpy() for x in train_ADE])
    # print('train FDE:', [x.detach().cpu().numpy() for x in train_FDE])

    train_ADE = torch.cat(train_ADE).mean()
    train_FDE = torch.cat(train_FDE).mean()

    # print('Total number of trajectories: {:d}'.format(counter))

    return train_ADE.item(), train_FDE.item(), train_loss.item()
