import re
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict, deque

from models.ynet import YNet
from utils.train_epoch import train_epoch
from utils.data_utils import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, get_patch, \
    preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from utils.evaluate import evaluate


def mark_encoder_bias_trainable(model):
    for param_name, param in model.encoder.named_parameters():
        if 'bias' in param_name: param.requires_grad = True
    return model 


def mark_goal_bias_trainable(model):
    for param_name, param in model.goal_decoder.named_parameters():
        if 'bias' in param_name: param.requires_grad = True
    return model 


def mark_traj_bias_trainable(model):
    for param_name, param in model.traj_decoder.named_parameters():
        if 'bias' in param_name: param.requires_grad = True
    return model 


def mark_ynet_bias_trainable(model):
    model = mark_encoder_bias_trainable(model)
    model = mark_goal_bias_trainable(model)
    model = mark_traj_bias_trainable(model)
    return model 


class YNetTrainer:
    def __init__(self, params, device=None):
        """
        Ynet class, following a sklearn similar class structure
        :param obs_len: observed timesteps
        :param pred_len: predicted timesteps
        :param params: dictionary with hyperparameters
        """
        self.params = params
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Working on {self.device}')

        self.division_factor = 2 ** len(params['encoder_channels'])
        self.template_size = int(4200 * params['resize_factor'])

        self.model = YNet(
            obs_len=params['obs_len'], pred_len=params['pred_len'],
            segmentation_model_fp=params['segmentation_model_fp'],
            use_features_only=params['use_features_only'],
            n_semantic_classes=params['n_semantic_classes'],
            encoder_channels=params['encoder_channels'],
            decoder_channels=params['decoder_channels'],
            n_waypoints=len(params['waypoints']),
            train_net=params['train_net'], 
            position=params['position'],
            network=params['network'],
            n_fusion=params['n_fusion']
        )
    
    def train(self, df_train, df_val, train_image_path, val_image_path, experiment_name):
        return self._train(df_train, df_val, train_image_path, val_image_path, experiment_name, **self.params)

    def _train(
        self, df_train, df_val, train_image_path, val_image_path, experiment_name, ckpt_path, 
        dataset_name, resize_factor, obs_len, pred_len, batch_size, lr, n_epoch, 
        waypoints, n_goal, n_traj, kernlen, nsig, e_unfreeze, loss_scale, temperature,
        use_raw_data=False, save_every_n=10, train_net="all", position=[], 
        fine_tune=False, augment=False, ynet_bias=False, 
        use_CWS=False, resl_thresh=0.002, CWS_params=None, n_early_stop=5, 
        steps=[20], lr_decay_ratio=0.1, network=None, swap_semantic=False, window_size=9, smooth_val=False, **kwargs):
        """
        Train function
        :param df_train: pd.df, train data
        :param df_val: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param train_image_path: str, filepath to train images
        :param val_image_path: str, filepath to val images
        :param experiment_name: str, arbitrary name to name weights file
        :param batch_size: int, batch size
        :param n_goal: int, number of goals per trajectory, K_e in paper
        :param n_traj: int, number of trajectory per goal, K_a in paper
        :return:
        """
        # get data
        train_images, train_loader, self.homo_mat = self.prepare_data(
            df_train, train_image_path, dataset_name, 'train', 
            obs_len, pred_len, resize_factor, use_raw_data, augment)
        val_images, val_loader, _ = self.prepare_data(
            df_val, val_image_path, dataset_name, 'val', 
            obs_len, pred_len, resize_factor, use_raw_data, False)

        # model 
        model = self.model.to(self.device)

        # Freeze segmentation model
        for param in model.semantic_segmentation.parameters():
            param.requires_grad = False

        if not (train_net == 'all' or train_net == 'train'):
            for param in model.parameters(): param.requires_grad = False

            # tune the whole encoder 
            if train_net == "encoder" and len(position) == 0:
                for param in model.encoder.parameters():
                    param.requires_grad = True
            # tune partial encoder 
            elif train_net == 'encoder' and len(position) > 0:
                for param_name, param in model.encoder.named_parameters():
                    param_layer = param_name.split('.')[1]
                    if param_layer in position: param.requires_grad = True
            # serial adapter
            elif 'serial' in train_net:
                for param_name, param in model.encoder.named_parameters():
                    if 'serial' in param_name: param.requires_grad = True
            # parallel adapter
            elif 'parallel' in train_net:
                for param_name, param in model.encoder.named_parameters():
                    if 'parallel' in param_name: param.requires_grad = True
            # mosa 
            elif 'mosa' in train_net:
                for param_name, param in model.encoder.named_parameters():
                    if 'lora' in param_name: param.requires_grad = True
            # after semantic segmentation 
            elif 'semantic' in train_net:
                for param_name, param in model.named_parameters():
                    if 'semantic_adapter' in param_name: param.requires_grad = True
            # tuned modified part 
            elif network == 'fusion' and train_net == 'scene':
                for param in model.encoder.scene_stages.parameters():
                    param.requires_grad = True
            elif network == 'fusion' and train_net == 'motion':
                for param in model.encoder.motion_stages.parameters():
                    param.requires_grad = True
            elif network == 'fusion' and train_net == 'scene_fusion':
                for param in model.encoder.scene_stages.parameters():
                    param.requires_grad = True
                for param in model.encoder.fusion_stages.parameters():
                    param.requires_grad = True
            elif network == 'fusion' and train_net == 'motion_fusion':
                for param in model.encoder.motion_stages.parameters():
                    param.requires_grad = True
                for param in model.encoder.fusion_stages.parameters():
                    param.requires_grad = True
            elif network == 'fusion' and train_net == 'scene_motion':
                for param in model.encoder.scene_stages.parameters():
                    param.requires_grad = True
                for param in model.encoder.motion_stages.parameters():
                    param.requires_grad = True
            elif network == 'fusion' and train_net == 'fusion':
                for param in model.encoder.fusion_stages.parameters():
                    param.requires_grad = True
            elif network == 'fusion' and train_net == 'scene_motion_fusion':
                for param in model.encoder.parameters():
                    param.requires_grad = True
            # bias term 
            elif train_net == 'biasEncoder':
                model = mark_encoder_bias_trainable(model)
            elif train_net == 'biasGoal':
                model = mark_goal_bias_trainable(model)
            elif train_net == 'biasTraj':
                model = mark_traj_bias_trainable(model)
            elif train_net == 'bias':
                model = mark_ynet_bias_trainable(model)
            # inside segmentation model 
            elif 'segmentation' in train_net:
                layer = train_net.split('_')[1]
                if layer in ['head', 'bias', 'bn']:
                    for param_name, param in model.semantic_segmentation.named_parameters():
                        if layer in param_name: param.requires_grad = True
                else:
                    for param_name, param in model.semantic_segmentation.named_parameters():
                        if re.search(f'decoder.blocks.\d.{layer}', param_name) is not None:
                            param.requires_grad = True 
            else:
                raise NotImplementedError
            # tuning all bias or not 
            if ynet_bias: 
                model = mark_ynet_bias_trainable(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if fine_tune:
            print("LR Schedular because finetuning")
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lr_decay_ratio)

        print('The number of trainable parameters: {:d}'.format(
            sum(param.numel() for param in model.parameters() if param.requires_grad)))

        criterion = nn.BCEWithLogitsLoss()

        # Create template
        input_template = torch.Tensor(create_dist_mat(size=self.template_size)).to(self.device)
        gt_template = torch.Tensor(create_gaussian_heatmap_template(
            size=self.template_size, kernlen=kernlen, nsig=nsig, normalize=False)).to(self.device)

        # train 
        best_val_ADE = 99999999999999
        best_epoch = 0
        self.val_ADE = []
        self.val_FDE = []
        state_dicts = deque()
        half_window_size = (window_size // 2) + 1

        print('Start training')
        for e in tqdm(range(n_epoch), desc='Epoch'):
            train_ADE, train_FDE, train_loss = train_epoch(
                model, train_loader, train_images, optimizer, criterion, loss_scale, self.device, 
                dataset_name, self.homo_mat, gt_template, input_template, waypoints,
                e, obs_len, pred_len, batch_size, e_unfreeze, resize_factor, 
                network, swap_semantic)

            # For faster inference, we don't use TTST and CWS here, only for the test set evaluation
            val_ADE, val_FDE, _, _ = evaluate(
                model, val_loader, val_images, self.device, 
                dataset_name, self.homo_mat, input_template, waypoints, 'val', 
                n_goal, n_traj, obs_len, batch_size, resize_factor,
                temperature, False, use_CWS, resl_thresh, CWS_params, 
                network=network, swap_semantic=swap_semantic)
            
            if fine_tune: 
                print(
                    f'Epoch {e}: 	Train (Top-1) ADE: {train_ADE:.2f} FDE: {train_FDE:.2f} 		Val (Top-k) ADE: {val_ADE:.2f} FDE: {val_FDE:.2f}   lr={lr_scheduler.get_last_lr()[0]}')
            else:
                print(
                    f'Epoch {e}: 	Train (Top-1) ADE: {train_ADE:.2f} FDE: {train_FDE:.2f} 		Val (Top-k) ADE: {val_ADE:.2f} FDE: {val_FDE:.2f}')
            self.val_ADE.append(val_ADE)
            self.val_FDE.append(val_FDE)
            if fine_tune:
                lr_scheduler.step()

            if smooth_val:
                # TODO: do not work if n_epoch < window_size
                print("Length: ", len(state_dicts))
                # Handle Model Ckpts Queue
                if len(state_dicts) == half_window_size:
                    curr_model_dict = state_dicts.popleft()
                    state_dicts.append(deepcopy(model.state_dict())) # Better to keep on CPU: TODO
                else:
                    state_dicts.append(deepcopy(model.state_dict())) # Better to keep on CPU: TODO
                # Handle smoothened ADE / FDE
                if e < window_size:
                    val_ADE = best_val_ADE + 1
                else:
                    val_ADE = sum(self.val_ADE[-window_size:])/window_size
            else:
                curr_model_dict = deepcopy(model.state_dict())
            
            if val_ADE < best_val_ADE:
                best_val_ADE = val_ADE
                best_epoch = e - half_window_size + 1 if smooth_val else e
                best_state_dict = curr_model_dict
                if not fine_tune:
                    print(f'Best Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
                    pathlib.Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(),  f'{ckpt_path}/{experiment_name}_weights.pt')

            if (e+1) % save_every_n == 0:
                pathlib.Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                self.save_params(f'{ckpt_path}/{experiment_name}__epoch_{e}.pt', train_net)

            # early stop in case of clear overfitting
            if fine_tune and (best_val_ADE < min(self.val_ADE[-n_early_stop:])):
                print(f'Early stop at epoch {e}')
                break

        # Load the best model
        print(f'Best epoch at {best_epoch}')
        if best_epoch != 0:
            model.load_state_dict(best_state_dict, strict=True)

        # Save the best model
        pathlib.Path(ckpt_path).mkdir(parents=True, exist_ok=True)
        pt_path = f'{ckpt_path}/{experiment_name}.pt'
        self.save_params(pt_path, train_net)

        return self.val_ADE, self.val_FDE

    def test(self, df_test, image_path, return_preds=False, return_samples=False):
        return self._test(df_test, image_path, 
            return_preds=return_preds, return_samples=return_samples, **self.params)

    def _test(
        self, df_test, image_path, dataset_name, resize_factor, 
        batch_size, n_round, obs_len, pred_len, 
        waypoints, n_goal, n_traj, temperature, rel_threshold, 
        use_TTST, use_CWS, CWS_params, use_raw_data=False, 
        return_preds=False, return_samples=False, network=None, swap_semantic=False, **kwargs):
        """
        Val function
        :param df_test: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param image_path: str, filepath to val images
        :param batch_size: int, batch size
        :param n_goal: int, number of goals per trajectory, K_e in paper
        :param n_traj: int, number of trajectory per goal, K_a in paper
        :param n_round: int, number of epochs to evaluate
        :return:
        """

        # get data 
        test_images, test_loader, self.homo_mat = self.prepare_data(
            df_test, image_path, dataset_name, 'test', 
            obs_len, pred_len, resize_factor, use_raw_data)

        # model 
        model = self.model.to(self.device)

        # Create template
        input_template = torch.Tensor(create_dist_mat(size=self.template_size)).to(self.device)

        self.eval_ADE = []
        self.eval_FDE = []
        list_metrics, list_trajs = [], []

        print("TTST setting:", use_TTST)
        print('Start testing')
        for e in tqdm(range(n_round), desc='Round'):
            test_ADE, test_FDE, df_metrics, trajs_dict = evaluate(
                model, test_loader, test_images, self.device, 
                dataset_name, self.homo_mat, input_template, waypoints, 'test',
                n_goal, n_traj, obs_len, batch_size, resize_factor,
                temperature, use_TTST, use_CWS, rel_threshold, CWS_params,
                return_preds=return_preds, return_samples=return_samples, 
                network=network, swap_semantic=swap_semantic)
            list_metrics.append(df_metrics)
            list_trajs.append(trajs_dict)
            print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')
            self.eval_ADE.append(test_ADE)
            self.eval_FDE.append(test_FDE)

        avg_ade = sum(self.eval_ADE) / len(self.eval_ADE)
        avg_fde = sum(self.eval_FDE) / len(self.eval_FDE)
        print(
            f'\nAverage performance (by {n_round}): \nTest ADE: {avg_ade} \nTest FDE: {avg_fde}')
        return avg_ade, avg_fde, list_metrics, list_trajs
    
    def forward_test(self, df_test, image_path, set_input, noisy_std_frac):
        return self._forward_test(df_test, image_path, set_input, noisy_std_frac, **self.params)

    def _forward_test(
        self, df_test, image_path, 
        set_input, noisy_std_frac, decision,
        dataset_name, obs_len, pred_len, resize_factor, 
        use_raw_data, waypoints, kernlen, nsig, loss_scale, **kwargs):

        # get data 
        test_images, test_loader, self.homo_mat = self.prepare_data(
            df_test, image_path, dataset_name, 'test', 
            obs_len, pred_len, resize_factor, use_raw_data)

        # create template
        input_template = torch.Tensor(create_dist_mat(size=self.template_size)).to(self.device)
        gt_template = torch.Tensor(create_gaussian_heatmap_template(
            size=self.template_size, kernlen=kernlen, nsig=nsig, normalize=False)).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        
        # test 
        if len(test_loader) == 1:
            for traj, _, scene_id in test_loader:
                scene_raw_img = test_images[scene_id].to(self.device).unsqueeze(0)
                if 'semantic' in set_input and (noisy_std_frac is not None): 
                    # noisy input
                    std = noisy_std_frac * (scene_raw_img.max() - scene_raw_img.min())
                    noisy_scene_img = scene_raw_img + scene_raw_img.new(scene_raw_img.size()).normal_(0, std)
                    noisy_scene_img.requires_grad = True
                    # forward 
                    if decision == 'loss':
                        goal_loss, traj_loss = self._forward_batch(
                            noisy_scene_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, False)
                    elif decision == 'map':
                        pred_goal_map, pred_traj_map = self._forward_batch(
                            noisy_scene_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, 
                            set_input, True)
                    else:
                        raise ValueError(f'No support for decision={decision}')
                elif ('semantic' not in set_input) and (noisy_std_frac is not None): 
                    # forward 
                    if decision == 'loss':
                        goal_loss, traj_loss = self._forward_batch(
                            noisy_scene_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, False)
                    elif decision == 'map':
                        pred_goal_map, pred_traj_map, semantic_input_cat = self._forward_batch(
                            scene_raw_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, 
                            set_input, noisy_std_frac, True)
                    else:
                        raise ValueError(f'No support for decision={decision}')
                else:  # noisy_std_frac is None
                    # require grad for input or not
                    scene_raw_img.requires_grad = False 
                    if 'scene' in set_input: scene_raw_img.requires_grad = True
                    # forward 
                    if decision == 'loss':
                        goal_loss, traj_loss = self._forward_batch(
                            scene_raw_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, False)
                    elif decision == 'map':
                        pred_goal_map, pred_traj_map = self._forward_batch(
                            scene_raw_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, 
                            set_input, noisy_std_frac, True)
                    else:
                        raise ValueError(f'No support for decision={decision}')
        elif len(test_loader) == 0:
            raise ValueError('No data is provided')
        else:
            # TODO: make it work for multiple scenes 
            raise ValueError(f'Received more than 1 scene ({len(test_loader)})')
        
        if decision == 'loss':
            if noisy_std_frac is not None:
                return goal_loss, traj_loss, scene_raw_img, noisy_scene_img
            else:
                return goal_loss, traj_loss, scene_raw_img 
        elif decision == 'map':
            if noisy_std_frac is not None:
                return pred_goal_map, pred_traj_map, scene_raw_img, noisy_scene_img, semantic_input_cat
            else:
                return pred_goal_map, pred_traj_map, scene_raw_img

    def _forward_batch(
        self, scene_raw_img, traj, input_template, gt_template, criterion, 
        obs_len, pred_len, waypoints, loss_scale, device, 
        set_input=None, noisy_std_frac=None, return_pred_map=False):
        
        # model 
        model = self.model.to(self.device)

        _, _, H, W = scene_raw_img.shape

        # create heatmap for observed trajectories 
        observed = traj[:, :obs_len, :].reshape(-1, 2).cpu().numpy() 
        observed_map = get_patch(input_template, observed, H, W)  
        observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W]) 

        # create heatmap for groundtruth future trajectories 
        gt_future = traj[:, obs_len:].to(device)
        gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
        gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])
        
        # create semantic segmentation map for all bacthes 
        scene_image = model.segmentation(scene_raw_img)
        semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)
        # possibly adapt semantic image 
        semantic_image = model.adapt_semantic(semantic_image)

        # forward 
        observed_map.requires_grad = False 
        noisy_semantic, noisy_traj = False, False 
        if ('semantic' in set_input) and (noisy_std_frac is not None):
            # noisy semantic
            std = noisy_std_frac * (semantic_image.max() - semantic_image.min())
            noisy_semantic_image = semantic_image + semantic_image.new(semantic_image.size()).normal_(0, std)
            noisy_semantic_image.requires_grad = True
            noisy_semantic = True 
        if ('traj' in set_input) and (noisy_std_frac is not None): 
            # noisy traj 
            std = noisy_std_frac * (observed_map.max() - observed_map.min())
            noisy_observed_map = observed_map + observed_map.new(observed_map.size()).normal_(0, std)
            noisy_observed_map.requires_grad = True
            noisy_traj = True 
        elif 'traj' in set_input:
            observed_map.requires_grad = True
        
        # feature input 
        if noisy_semantic and noisy_traj:
            features = model.pred_features(noisy_semantic_image, noisy_semantic_image)
        elif noisy_semantic and not noisy_traj:
            features = model.pred_features(noisy_semantic_image, observed_map)
        elif not noisy_semantic and noisy_traj:
            features = model.pred_features(semantic_image, noisy_observed_map)
        else:
            features = model.pred_features(semantic_image, observed_map)
        pred_goal_map = model.pred_goal(features)

        # goal loss 
        goal_loss = criterion(pred_goal_map, gt_future_map) * loss_scale  
        pred_waypoint_map = pred_goal_map[:, waypoints] 
        
        # way points 
        pred_waypoints_maps_downsampled = [nn.AvgPool2d(
            kernel_size=2**i, stride=2**i)(pred_waypoint_map) for i in range(1, len(features))]
        pred_waypoints_maps_downsampled = [pred_waypoint_map] + pred_waypoints_maps_downsampled
        
        # traj loss
        traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(
            features, pred_waypoints_maps_downsampled)]
        pred_traj_map = model.pred_traj(traj_input)
        traj_loss = criterion(pred_traj_map, gt_future_map) * loss_scale  
        
        if return_pred_map:
            if noisy_semantic or noisy_traj:
                return pred_goal_map, pred_traj_map, torch.cat([semantic_image, observed_map], dim=1)
            else:
                return pred_goal_map, pred_traj_map
        return goal_loss, traj_loss

    def prepare_data(
        self, df, image_path, dataset_name, mode, obs_len, pred_len, 
        resize_factor, use_raw_data, augment=False):
        """
        Prepare dataset for training, validation, and testing. 

        Args:
            df (pd.DataFrame): df_train / df_val / df_test 
            image_path (str): path storing scene images 
            dataset_name (str): name of the dataset
            mode (str): choices=[train, val, test]
            resize_factor (float): _description_
            use_raw_data (bool): _description_
            fine_tune (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # get image filename 
        dataset_name = dataset_name.lower()
        if dataset_name == 'sdd':
            image_file_name = 'reference.jpg'
        elif dataset_name == 'ind-dataset-v1.0':
            image_file_name = 'reference.png'
        elif dataset_name == 'eth':
            image_file_name = 'oracle.png'
        else:
            raise ValueError(f'{dataset_name} dataset is not supported') 

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(self.device)
            seg_mask = True
        else:
            homo_mat = None
            seg_mask = False
        # Load scene images 
        if not augment:
            # do not augment train data and images 
            images_dict = create_images_dict(
                df.sceneId.unique(), image_path=image_path, 
                image_file=image_file_name, use_raw_data=use_raw_data)
            print('No data and images augmentation')
        else: 
            # augment train data and images
            df, images_dict = augment_data(
                df, image_path=image_path, image_file=image_file_name,
                seg_mask=seg_mask, use_raw_data=use_raw_data)
            print('Augmented data and images')

        # Initialize dataloaders
        dataset = SceneDataset(df, resize=resize_factor, total_len=obs_len+pred_len)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=scene_collate, 
            shuffle=True if mode=='train' else False)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(images_dict, factor=resize_factor, seg_mask=seg_mask)
        # make sure that image shape is divisible by 32, for UNet segmentation
        pad(images_dict, division_factor=self.division_factor)
        preprocess_image_for_segmentation(images_dict, seg_mask=seg_mask)

        return images_dict, dataloader, homo_mat

    def load_params(self, path):
        if self.device == torch.device('cuda'):
            self.model.load_state_dict(torch.load(path), strict=False)
            print('Loaded ynet model to GPU')
        else:  # self.device == torch.device('cpu')
            self.model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
            print('Loaded ynet model to CPU')

    def save_params(self, path, train_net):
        if train_net == 'all' or train_net == 'train':
            state_dict = self.model.state_dict()
            state_dict = {k:v for k,v in state_dict.items() if 'segmentation' not in k}
        else:
            # save parameters with requires_grad = True
            state_dict = OrderedDict()
            for param_name, param in self.model.named_parameters():
                if param.requires_grad:
                    state_dict[param_name] = param 
        torch.save(state_dict, path)

    def load_separated_params(self, pretrained_path, tuned_path):
        if self.device == torch.device('cuda'):
            self.model.load_state_dict(torch.load(pretrained_path), strict=False)
            self.model.load_state_dict(torch.load(tuned_path), strict=False)
            print('Loaded ynet model to GPU')
        else:  # self.device == torch.device('cpu')
            self.model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
            self.model.load_state_dict(torch.load(tuned_path, map_location='cpu'), strict=False)
            print('Loaded ynet model to CPU')
        