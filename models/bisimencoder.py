import torch
import torch.nn as nn
from torch.nn import functional as F

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class StateBeliefEncoder(nn.Module):    #  Encoder for SAC
    def __init__(self,augmented_obs_dim,feature_dim,max_norm,layers):  # eg. layers=[128,64,50]
        super().__init__()
        self.output_size = feature_dim
        self.num_layers=len(layers)
        curr_input_size=augmented_obs_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
            curr_input_size = layers[i]

        self.fc_out = nn.Linear(curr_input_size, self.output_size)

        self.max_norm=max_norm


    def forward(self, augmented_obs, detach=False,normalize=True):

        h = augmented_obs

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        h=self.fc_out(h)
        
        if self.max_norm and normalize:
            h = self.normalize(h)

        return h
    
    def normalize(self,x):
        if self.max_norm:
            norms = x.norm(dim=-1)
            norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
            x = x / norm_to_max

        return x

    # def copy_conv_weights_from(self, source):
    #     """Tie convolutional layers"""
    #     # only tie conv layers
    #     for i in range(self.num_layers):
    #         tie_weights(src=source.fc_layers[i], trg=self.fc_layers[i])
    def copy_conv_weights_from(self, source):
        source_layers = [m for m in source.modules() if isinstance(m, nn.Linear)]
        self_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        assert (len(self_layers) == len(source_layers))
        for self_layer, source_layer in zip(self_layers, source_layers):
            tie_weights(src=source_layer, trg=self_layer)


class StateEncoder(nn.Module):   # only encode state
    def __init__(self,obs_dim,feature_dim,max_norm,layers): 
        super().__init__()
        self.output_size = feature_dim
        self.num_layers=len(layers)
        curr_input_size=obs_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
            curr_input_size = layers[i]
        self.fc_out = nn.Linear(curr_input_size, self.output_size)
        # self.max_norm=5
        self.max_norm=None

    def forward(self, obs, detach=False,normalize=False):

        h = obs

        for i in range(len(self.fc_layers)):
            h = F.elu(self.fc_layers[i](h))
        h=self.fc_out(h)
        
        if self.max_norm and normalize:
            h = self.normalize(h)
        return h
    
    def normalize(self,x):
        if self.max_norm:
            norms = x.norm(dim=-1)
            norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
            x = x / norm_to_max

        return x




class MLPEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim,  *args, **kwargs):
        super().__init__()
        
        obs_shape = obs_shape
        self.model = nn.Sequential(
            nn.Linear(obs_shape, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.feature_dim = feature_dim
        self.max_norm = kwargs.get("max_norm")

    def forward(self, obs, detach=False, normalize=True):
        x = self.model(obs)
        if self.max_norm and normalize:
            x = self.normalize(x)

        if detach:
            x = x.detach()

        return x

    def normalize(self, x):
        if self.max_norm:
            norms = x.norm(dim=-1)
            norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
            x = x / norm_to_max

        return x

    def copy_conv_weights_from(self, source):
        source_layers = [m for m in source.modules() if isinstance(m, nn.Linear)]
        self_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        assert (len(self_layers) == len(source_layers))
        for self_layer, source_layer in zip(self_layers, source_layers):
            tie_weights(src=source_layer, trg=self_layer)



class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        out_dims = 100  # if defaults change, adjust this as needed
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        out_dims = 56  # 3 cameras
        # out_dims = 100  # 5 cameras
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


                        
_AVAILABLE_ENCODERS = {'mlp': MLPEncoder,
                       'statebelief':StateBeliefEncoder,
                       'state_encoder':StateEncoder,
                       'pixel': PixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim,max_norm=None,layers=None
):
    assert encoder_type in _AVAILABLE_ENCODERS

    if encoder_type == 'mlp':
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape,feature_dim,max_norm=max_norm,layers=None
        )
    elif encoder_type == 'statebelief':
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape,feature_dim,max_norm=max_norm,layers=layers
        )
    elif encoder_type == 'state_encoder':
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape,feature_dim,max_norm=max_norm,layers=layers
        )
    else:

        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, layers
        )

















