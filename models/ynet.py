import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from utils.softargmax import SoftArgmax2D, create_meshgrid


def conv2d(in_channels, out_channels=None, kernel_size=1, stride=1, padding=None, is_bias=False):
    if padding is None: padding = kernel_size // 2
    if out_channels is None: out_channels = in_channels
    return nn.Conv2d(in_channels, out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding, bias=is_bias)


class Adapter(nn.Module):
    def __init__(
        self, adapter_name, in_channels, out_channels=None, stride=1, is_bias=False):
        super(Adapter, self).__init__()
        self.is_bias = is_bias
        self.adapter_name = adapter_name
        self.adapter_size = adapter_name.split('_')[1:]
        self.is_multiple = False if len(self.adapter_size) < 2 else True
        # default serial adapter
        if 'serial' in self.adapter_name:
            self.serial_layer = nn.Sequential(
                nn.BatchNorm2d(in_channels), conv2d(in_channels, is_bias=is_bias))
        # parallel adapter 
        elif 'parallel' in self.adapter_name and not self.is_multiple:
            kernel_size = int(self.adapter_size[0].split('x')[0]) if self.adapter_size else 1
            self.parallel_layer = conv2d(
                in_channels, out_channels, kernel_size, stride, is_bias=is_bias)
        # multiple parallel adapter 
        elif 'parallel' in self.adapter_name and self.is_multiple:
            self.parallel_layer = nn.ModuleList()
            for adapter_size in self.adapter_size:
                kernel_size = int(adapter_size.split('x')[0])
                self.parallel_layer.append(
                    conv2d(in_channels, out_channels, kernel_size, stride, is_bias=is_bias))
        else:
            raise ValueError(f'Invalid adapter={self.adapter_name}')
        
        # initialize parameters
        self.initialize()

    def initialize(self):
        if 'serial' in self.adapter_name:
            nn.init.zeros_(self.serial_layer[1].weight)
            if self.is_bias: nn.init.zeros_(self.serial_layer[1].is_bias)
        elif 'parallel' in self.adapter_name:
            for p in self.parallel_layer.parameters():
                nn.init.zeros_(p) 


class AdapterBlock(Adapter):
    def forward(self, x):
        if 'parallel' in self.adapter_name:
            if self.is_multiple:
                y = 0
                for layer in self.parallel_layer:
                    x_ = layer(x)
                    y += x_
            else:
                y = self.parallel_layer(x)
        elif 'serial' in self.adapter_name:
            y = self.serial_layer(x)
            y += x
        return y


class AdapterLayer(nn.Conv2d):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        adapter_name: str, 
        adapter_dropout: float = 0.,
        stride: int = 1,
        is_bias: bool = False,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        self.is_bias = is_bias
        self.adapter_name = adapter_name
        self.adapter_size = adapter_name.split('_')[1:]
        self.is_multiple = False if len(self.adapter_size) < 2 else True
        # default serial adapter
        if 'serial' in self.adapter_name:
            self.serial_layer = nn.Sequential(
                nn.BatchNorm2d(out_channels), conv2d(out_channels, is_bias=is_bias))
        # parallel adapter 
        elif 'parallel' in self.adapter_name and not self.is_multiple:
            kernel_size = int(self.adapter_size[0].split('x')[0]) if self.adapter_size else 1
            self.parallel_layer = conv2d(
                in_channels, out_channels, kernel_size, stride, is_bias=is_bias)
        # multiple parallel adapter 
        elif 'parallel' in self.adapter_name and self.is_multiple:
            self.parallel_layer = nn.ModuleList()
            for adapter_size in self.adapter_size:
                kernel_size = int(adapter_size.split('x')[0])
                self.parallel_layer.append(
                    conv2d(in_channels, out_channels, kernel_size, stride, is_bias=is_bias))
        else:
            raise ValueError(f'Invalid adapter={self.adapter_name}')
        
        # initialize parameters
        self.initialize()

    def initialize(self):
        if 'serial' in self.adapter_name:
            nn.init.zeros_(self.serial_layer[1].weight)
            if self.is_bias: nn.init.zeros_(self.serial_layer[1].is_bias)
        elif 'parallel' in self.adapter_name:
            for p in self.parallel_layer.parameters():
                nn.init.zeros_(p) 

    def forward(self, x):
        out = nn.Conv2d.forward(self, x)
        if 'serial' in self.adapter_name:
            y = self.serial_layer(out)
            y += out
        elif 'parallel' in self.adapter_name:
            if self.is_multiple:
                y = 0
                for layer in self.parallel_layer:
                    x_ = layer(x)
                    y += x_
            else:
                y = self.parallel_layer(x)
            y += out 
        return y


def get_conv2d(
    train_net, l, position, kernel_size, in_channels, 
    out_channels=None, rank=None, stride=1, padding=None):
    if out_channels is None: out_channels = in_channels
    if padding is None: padding = kernel_size // 2
    l = str(l)
    position = [str(i) for i in position]
    if 'mosa' in train_net and l in position:
        assert rank != 0 and rank is not None
        return lora.Conv2d(in_channels, out_channels, 
            kernel_size=kernel_size, r=rank, stride=stride, padding=padding)
    elif 'Layer' in train_net and l in position:
        return AdapterLayer(adapter_name=train_net, 
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding)
    else:
        return nn.Conv2d(in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding)


class Embedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        return self.conv(x)


class YNetEncoder(nn.Module):
    def __init__(
        self, in_channels, channels=(64, 128, 256, 512, 512), 
        train_net=None, position=[]):
        """
        Encoder model
        :param in_channels: int, n_semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super(YNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = channels
        self.train_net = train_net
        self.position = position
        self.stages = nn.ModuleList()
        if 'mosa' in self.train_net: 
            self.rank = int(self.train_net.split('_')[1]) if len(self.train_net.split('_')) > 1 else 1
        else:
            self.rank = None 
        
        # First block
        modules = [
            get_conv2d(
                train_net=train_net, l=0, position=position, kernel_size=3, 
                in_channels=in_channels, out_channels=channels[0], rank=self.rank),
            nn.ReLU(inplace=False)]
        self.stages.append(nn.Sequential(*modules))

        # Subsequent blocks, each starts with MaxPool
        for i in range(len(channels) - 1):
            modules = [
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
                get_conv2d(
                    train_net=train_net, l=i+1, position=position, 
                    kernel_size=3, in_channels=channels[i], out_channels=channels[i+1], rank=self.rank), 
                nn.ReLU(inplace=False), 
                get_conv2d(
                    train_net=train_net, l=i+1, position=position, 
                    kernel_size=3, in_channels=channels[i+1], out_channels=channels[i+1], rank=self.rank), 
                nn.ReLU(inplace=False)]
            self.stages.append(nn.Sequential(*modules))
        
        # Last MaxPool layer before passing the features into decoder
        self.stages.append(nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))


class YNetEncoderL(YNetEncoder):
    def __init__(
        self, in_channels, channels=(64, 128, 256, 512, 512), 
        train_net=None, position=[]):
        """
        Encoder model
        :param in_channels: int, n_semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super(YNetEncoderL, self).__init__(in_channels, channels, train_net, position)

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features 


class YNetEncoderB(YNetEncoder):
    def __init__(
        self, in_channels, channels=(64, 128, 256, 512, 512), 
        train_net=None, position=[]):
        """
        Encoder model
        :param in_channels: int, n_semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        self.position = [int(i) for i in position]
        super(YNetEncoderB, self).__init__(in_channels, channels, train_net, self.position)

        # adapter
        par_channels_in = [in_channels] + channels[:-1]
        if 'serial' in self.train_net:
            self.adapters = nn.ModuleList([
                AdapterBlock(train_net, channels[i]) for i in self.position])
        elif 'parallel' in self.train_net:
            self.adapters = nn.ModuleList([
                AdapterBlock(train_net, par_channels_in[i], channels[i]) for i in self.position])

    def forward(self, x):
        features = []
        j = 0
        for i, stage in enumerate(self.stages):
            if 'serial' in self.train_net:
                x = stage(x) 
                if i in self.position:
                    x = self.adapters[j](x)
                    j += 1
            elif 'parallel' in self.train_net:
                if isinstance(stage[0], nn.MaxPool2d):
                    y = stage[0](x)
                    x = stage(x)
                    if i in self.position:
                        x = x + self.adapters[j](y)
                        j += 1
                else:
                    y = stage(x)
                    if i in self.position:
                        y = y + self.adapters[j](x)
                        j += 1
                    x = y
            else:
                x = stage(x)
            features.append(x)
        return features 


class YNetEncoderFusion(nn.Module):
    def __init__(
        self, scene_channel, motion_channel, channels,
        train_net=None, position=[], n_fusion=2):
        super().__init__()
        self.scene_channel = scene_channel
        self.motion_channel = motion_channel
        self.channels = channels
        self.train_net = train_net
        self.position = position
        
        if 'mosa' in self.train_net: 
            self.rank = int(self.train_net.split('_')[1]) if len(self.train_net.split('_')) > 1 else 1
        else:
            self.rank = None 

        # check channels are even 
        assert not any([i%2 for i in channels]), f'Odd value in channels={channels}'
        assert n_fusion <= len(channels) - 1, f'The number of fusion exceeds the total number of layer in encoder'

        self.scene_stages = nn.ModuleList([
            nn.Sequential(
                get_conv2d(
                    train_net=train_net, l='scene', position=position, kernel_size=3, 
                    in_channels=scene_channel, out_channels=channels[0]//2, rank=self.rank),
                nn.ReLU(inplace=False))
        ])
        self.motion_stages = nn.ModuleList([
            nn.Sequential(
                get_conv2d(
                    train_net=train_net, l='motion', position=position, kernel_size=3, 
                    in_channels=motion_channel, out_channels=channels[0]//2, rank=self.rank),
                nn.ReLU(inplace=False))
        ])
        self.fusion_stages = nn.ModuleList()

        n_sep = len(channels) - n_fusion - 1
        # scene part 
        for i in range(n_sep):
            modules = [
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
                get_conv2d(
                    train_net=train_net, l='scene', position=position, 
                    kernel_size=3, in_channels=channels[i]//2, out_channels=channels[i+1]//2, rank=self.rank), 
                nn.ReLU(inplace=False), 
                get_conv2d(
                    train_net=train_net, l='scene', position=position, 
                    kernel_size=3, in_channels=channels[i+1]//2, out_channels=channels[i+1]//2, rank=self.rank), 
                nn.ReLU(inplace=False)]
            self.scene_stages.append(nn.Sequential(*modules))

        # motion part 
        for i in range(n_sep):
            modules = [
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
                get_conv2d(
                    train_net=train_net, l='motion', position=position, 
                    kernel_size=3, in_channels=channels[i]//2, out_channels=channels[i+1]//2, rank=self.rank), 
                nn.ReLU(inplace=False), 
                get_conv2d(
                    train_net=train_net, l='motion', position=position, 
                    kernel_size=3, in_channels=channels[i+1]//2, out_channels=channels[i+1]//2, rank=self.rank), 
                nn.ReLU(inplace=False)]
            self.motion_stages.append(nn.Sequential(*modules))
        
        # fusion part
        for i in range(n_sep, len(channels) - 1):
            modules = [
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
                get_conv2d(
                    train_net=train_net, l='fusion', position=position, 
                    kernel_size=3, in_channels=channels[i], out_channels=channels[i+1], rank=self.rank), 
                nn.ReLU(inplace=False), 
                get_conv2d(
                    train_net=train_net, l='fusion', position=position, 
                    kernel_size=3, in_channels=channels[i+1], out_channels=channels[i+1], rank=self.rank), 
                nn.ReLU(inplace=False)]
            self.fusion_stages.append(nn.Sequential(*modules))

        # Last MaxPool layer before passing the features into decoder
        self.fusion_stages.append(nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

    def forward(self, scene_map, motion_map):
        # scene part
        scene_features = []
        x = scene_map
        for stage in self.scene_stages:
            x = stage(x)
            scene_features.append(x)
        
        # motion part
        motion_features = []
        x = motion_map 
        for stage in self.motion_stages:
            x = stage(x)
            motion_features.append(x)

        # concatenate scene and motion 
        features = []
        for scene_feature, motion_feature in zip(scene_features, motion_features):
            features.append(torch.cat([scene_feature, motion_feature], axis=1))

        # fusion part
        x = features[-1]
        for stage in self.fusion_stages:
            x = stage(x)
            features.append(x)
        
        return features


class YNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
        """
        Decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
        """
        super(YNetDecoder, self).__init__()

        # The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
        if traj:
            encoder_channels = [channel + traj for channel in encoder_channels]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        center_channels = encoder_channels[0]

        decoder_channels = decoder_channels

        # The center layer (the layer with the smallest feature map size)
        self.center = nn.Sequential(
            nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False)
        )

        # Determine the upsample channel dimensions
        upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
        upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

        # Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
        self.upsample_conv = nn.ModuleList([
            nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
                for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)])

        # Determine the input and output channel dimensions of each layer in the decoder
        # As we concat the encoded feature and decoded features we have to sum both dims
        in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
        out_channels = decoder_channels

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=False),
                nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=False))
            for in_channels_, out_channels_ in zip(in_channels, out_channels)]
        )

        # Final 1x1 Conv prediction to get our heatmap logits (before softmax)
        self.predictor = nn.Conv2d(
            in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        # Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
        # reverse the order of encoded features, as the decoder starts from the smallest image
        features = features[::-1]
        # decoder: layer 1
        center_feature = features[0]
        x = self.center(center_feature)
        # decoder: layer 2-6
        for f, d, up in zip(features[1:], self.decoder, self.upsample_conv):
            # bilinear interpolation for upsampling
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = up(x)  # 3x3 conv for upsampling
            # concat encoder and decoder features
            x = torch.cat([x, f], dim=1)
            x = d(x)
        # decoder: layer 7 (last predictor layer)
        x = self.predictor(x) 
    
        return x


class YNet(nn.Module):
    def __init__(
        self, obs_len, pred_len, segmentation_model_fp, 
        use_features_only=False, n_semantic_classes=6,
        encoder_channels=[], decoder_channels=[], n_waypoints=1, 
        train_net=None, position=[], network=None, n_fusion=None):
        """
        Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
        :param obs_len: int, observed timesteps
        :param pred_len: int, predicted timesteps
        :param segmentation_model_fp: str, filepath to pretrained segmentation model
        :param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
        :param n_semantic_classes: int, number of semantic classes
        :param encoder_channels: list, encoder channel structure
        :param decoder_channels: list, decoder channel structure
        :param n_waypoints: int, number of waypoints
        """
        super(YNet, self).__init__()

        self.train_net = train_net 

        if segmentation_model_fp is not None:
            if torch.cuda.is_available():
                self.semantic_segmentation = torch.load(segmentation_model_fp)
                print('Loaded segmentation model to GPU')
            else:
                self.semantic_segmentation = torch.load(
                    segmentation_model_fp, map_location=torch.device('cpu'))
                print('Loaded segmentation model to CPU')
            if use_features_only:
                self.semantic_segmentation.segmentation_head = nn.Identity()
                n_semantic_classes = 16  # instead of classes use number of feature_dim
        else:
            self.semantic_segmentation = nn.Identity()
        
        self.feature_channels = n_semantic_classes + obs_len
        self.network = network

        # semantic tuning 
        if 'semantic' in train_net:
            kernel_size = int(train_net.split('_')[-1].split('x')[0])
            self.semantic_adapter = get_conv2d(
                train_net=train_net, l=None, position=None, kernel_size=kernel_size, 
                in_channels=n_semantic_classes, out_channels=n_semantic_classes)
            nn.init.zeros_(self.semantic_adapter.weight)
            nn.init.zeros_(self.semantic_adapter.bias)
        
        if network == 'fusion':
            assert n_fusion is not None
            self.encoder = YNetEncoderFusion(
                scene_channel=n_semantic_classes, motion_channel=obs_len, channels=encoder_channels,
                train_net=train_net, position=position, n_fusion=n_fusion
            )
        elif network == 'original' or network == 'embed':
            # adding embedding layer before concatenation
            if network == 'embed':
                self.scene_embedding = Embedding(n_semantic_classes)
                self.motion_embedding = Embedding(obs_len)

            if 'mosa' in train_net or 'Layer' in train_net:
                self.encoder = YNetEncoderL(
                    in_channels=self.feature_channels, channels=encoder_channels,
                    train_net=train_net, position=position)
            else:
                self.encoder = YNetEncoderB(
                    in_channels=self.feature_channels, channels=encoder_channels,
                    train_net=train_net, position=position)
        else:
            raise ValueError('No network parameter is provided')

        self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
        self.traj_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=n_waypoints)

        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)

        self.encoder_channels = encoder_channels

    def segmentation(self, image):
        return self.semantic_segmentation(image)

    def adapt_semantic(self, semantic_img):
        if 'semantic' in self.train_net:
            semantic_adapted = self.semantic_adapter(semantic_img)
            return semantic_adapted + semantic_img
        else:
            return semantic_img

    # Forward pass for goal decoder
    def pred_goal(self, features):
        return self.goal_decoder(features)

    # Forward pass for trajectory decoder
    def pred_traj(self, features):
        return self.traj_decoder(features)

    # Forward pass for feature encoder, returns list of feature maps
    def pred_features(self, scene_map, motion_map):
        if self.network == 'fusion':
            return self.encoder(scene_map, motion_map)
        else:   
            x = torch.cat([scene_map, motion_map], dim=1) 
            return self.encoder(x)

    # Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
    def softmax(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

    # Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
    def softargmax(self, output):
        return self.softargmax_(output)

    def sigmoid(self, output):
        return torch.sigmoid(output)

    def softargmax_on_softmax_map(self, x):
        """ Softargmax: As input a batched image where softmax is already performed (not logits) """
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        x = x.flatten(2)

        estimated_x = pos_x * x
        estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
        estimated_y = pos_y * x
        estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
        softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
        return softargmax_coords
