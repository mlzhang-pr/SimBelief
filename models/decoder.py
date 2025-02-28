
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class StateTransitionDecoder(nn.Module):
    def __init__(self,
                 task_embedding_size,
                 layers,
                 #
                 action_size,
                 action_embed_size,
                 state_size,
                 state_embed_size,
                #  pred_type='deterministic'  # 修改为高斯
                 pred_type='gaussian'
                 ):
        super(StateTransitionDecoder, self).__init__()

        self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)

        curr_input_size = task_embedding_size + state_embed_size + action_embed_size
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
            curr_input_size = layers[i]

        # output layer
        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_size, 2 * state_size)
        else:
            self.fc_out = nn.Linear(curr_input_size, state_size)

    def forward(self, task_embedding, state, action):

        ha = self.action_encoder(action)
        hs = self.state_encoder(state)
        h = torch.cat((task_embedding, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class StateTransitionDecoderBisim(nn.Module):
    def __init__(self,
                 task_embedding_size,
                 layers,
                 #
                 action_size,
                 action_embed_size,
                 augmented_encode_state_size,
                 state_embed_size,
                #  pred_type='deterministic'  # 修改为高斯
                 pred_type='gaussian'
                 ):
        super(StateTransitionDecoderBisim, self).__init__()
        self.pred_type=pred_type
        self.state_encoder = utl.FeatureExtractor(augmented_encode_state_size, state_embed_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)

        self.max_sigma=1e1
        self.min_sigma=1e-4

        curr_input_size = state_embed_size + action_embed_size
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
            curr_input_size = layers[i]

        self.ln = nn.LayerNorm(curr_input_size)
        
        # output layer
        if pred_type == 'gaussian':
            self.fc_mu = nn.Linear(curr_input_size, 50)
            self.fc_sigma = nn.Linear(curr_input_size, 50)
        else:
            self.fc_mu = nn.Linear(curr_input_size, augmented_encode_state_size)

    def forward(self, augmented_encode_state, action):

        ha = self.action_encoder(action)
        hs = self.state_encoder(augmented_encode_state)
        h = torch.cat((hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        h=self.ln(h)
        if self.pred_type == 'gaussian':
            mu = self.fc_mu(h)
            sigma = torch.sigmoid(self.fc_sigma(h))  # range (0, 1.)
            sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
            return mu,sigma
        else:
            mu = self.fc_mu(h)
            return mu



class RewardDecoder(nn.Module):
    def __init__(self,
                 layers,
                 task_embedding_size,
                 action_size,
                 action_embed_size,
                 state_size,
                 state_embed_size,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=False,
                 input_action=True,
                 ):
        super(RewardDecoder, self).__init__()

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        if self.multi_head:
            # one output head per state to predict rewards
            curr_input_size = task_embedding_size
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
                curr_input_size = layers[i]
            self.fc_out = nn.Linear(curr_input_size, num_states)
        else:
            # get state as input and predict reward prob
            self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
            self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)
            curr_input_size = task_embedding_size + state_embed_size
            if input_prev_state:
                curr_input_size += state_embed_size
            if input_action:
                curr_input_size += action_embed_size
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
                curr_input_size = layers[i]

            if pred_type == 'gaussian':
                self.fc_out = nn.Linear(curr_input_size, 2)
            else:
                self.fc_out = nn.Linear(curr_input_size, 1)

    def forward(self, task_embedding, next_state, prev_state=None, action=None):

        if self.multi_head:
            h = task_embedding
        if not self.multi_head:
            # task_embedding = task_embedding.reshape((-1, task_embedding.shape[-1]))
            # next_state = next_state.reshape((-1, next_state.shape[-1]))
            hns = self.state_encoder(next_state)
            h = torch.cat((task_embedding, hns), dim=-1)
            if self.input_action:
                # action = action.reshape((-1, action.shape[-1]))
                ha = self.action_encoder(action)
                h = torch.cat((h, ha), dim=-1)
            if self.input_prev_state:
                # prev_state = prev_state.reshape((-1, prev_state.shape[-1]))
                hps = self.state_encoder(prev_state)
                h = torch.cat((h, hps), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        p_x = self.fc_out(h)
        if self.pred_type == 'deterministic' or self.pred_type == 'gaussian':
            pass
        elif self.pred_type == 'bernoulli':
            p_x = torch.sigmoid(p_x)
        elif self.pred_type == 'categorical':
            p_x = torch.softmax(p_x, 1)
        else:
            raise NotImplementedError

        return p_x


class RewardDecoderBisim(nn.Module):
    def __init__(self,
                 layers,
                 task_embedding_size,
                 action_size,
                 action_embed_size,
                 augmented_encode_state_size,
                 state_embed_size,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=False,  # defalt = True
                 input_action=True,
                 ):
        super(RewardDecoderBisim, self).__init__()

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        if self.multi_head:
            # one output head per state to predict rewards
            curr_input_size = task_embedding_size
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
                curr_input_size = layers[i]
            self.fc_out = nn.Linear(curr_input_size, num_states)
        else:
            # get state as input and predict reward prob
            self.state_encoder = utl.FeatureExtractor(augmented_encode_state_size, state_embed_size, F.relu)
            self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)
            curr_input_size =  state_embed_size
            if input_prev_state:
                curr_input_size += state_embed_size
            if input_action:
                curr_input_size += action_embed_size
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
                curr_input_size = layers[i]

            if pred_type == 'gaussian':
                self.fc_out = nn.Linear(curr_input_size, 2)
            else:
                self.fc_out = nn.Linear(curr_input_size, 1)

    def forward(self, augmented_encode_state, prev_state=None, action=None):   # augmented_encode_state=concat[phi(s),z]

        if not self.multi_head:
            # task_embedding = task_embedding.reshape((-1, task_embedding.shape[-1]))
            # next_state = next_state.reshape((-1, next_state.shape[-1]))
            h = self.state_encoder(augmented_encode_state)
            # h = torch.cat((task_embedding, hns), dim=-1)
            if self.input_action:
                # action = action.reshape((-1, action.shape[-1]))
                ha = self.action_encoder(action)
                h = torch.cat((h, ha), dim=-1)
            if self.input_prev_state:
                # prev_state = prev_state.reshape((-1, prev_state.shape[-1]))
                hps = self.state_encoder(prev_state)
                h = torch.cat((h, hps), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        p_x = self.fc_out(h)
        if self.pred_type == 'deterministic' or self.pred_type == 'gaussian':
            pass
        elif self.pred_type == 'bernoulli':
            p_x = torch.sigmoid(p_x)
        elif self.pred_type == 'categorical':
            p_x = torch.softmax(p_x, 1)
        else:
            raise NotImplementedError

        return p_x



# class RewardDecoderBisim(nn.Module):
#     def __init__(self,
#                  layers,
#                  task_embedding_size,
#                  action_size,
#                  action_embed_size,
#                  state_size,
#                  state_embed_size,
#                  num_states,
#                  statebelief_encoder=None,
#                  multi_head=False,
#                  pred_type='deterministic',
#                  input_prev_state=True,
#                  input_action=True,
                 
#                  ):
#         super(RewardDecoderBisim, self).__init__()

#         self.pred_type = pred_type
#         self.multi_head = multi_head
#         self.input_prev_state = input_prev_state
#         self.input_action = input_action

#         # self.statebelief_encoder=statebelief_encoder

#         if self.multi_head:
#             # one output head per state to predict rewards
#             curr_input_size = task_embedding_size
#             self.fc_layers = nn.ModuleList([])
#             for i in range(len(layers)):
#                 self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
#                 curr_input_size = layers[i]
#             self.fc_out = nn.Linear(curr_input_size, num_states)
#         else:
#             # get state as input and predict reward prob
#             # self.state_encoder1 = utl.FeatureExtractor(50, state_embed_size, F.relu)   # hidden_augmented_obs no need to pass featureextractor
#             self.state_encoder2 = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
#             self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)
#             curr_input_size = 50
#             if input_prev_state:
#                 curr_input_size += state_embed_size
#             if input_action:
#                 curr_input_size += action_embed_size
#             self.fc_layers = nn.ModuleList([])
#             for i in range(len(layers)):
#                 self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
#                 curr_input_size = layers[i]

#             if pred_type == 'gaussian':
#                 self.fc_out = nn.Linear(curr_input_size, 2)
#             else:
#                 self.fc_out = nn.Linear(curr_input_size, 1)

#     def forward(self, hidden_augmented_next_state, prev_state=None, action=None):

#         # if self.multi_head:
#         #     h = task_embedding
#         if not self.multi_head:
            
#             # h = self.state_encoder1(hidden_augmented_next_state)
#             h = hidden_augmented_next_state
#             if self.input_action:
#                 # action = action.reshape((-1, action.shape[-1]))
#                 ha = self.action_encoder(action)
#                 h = torch.cat((h, ha), dim=-1)
#             if self.input_prev_state:
#                 # prev_state = prev_state.reshape((-1, prev_state.shape[-1]))
#                 hps = self.state_encoder2(prev_state)
#                 h = torch.cat((h, hps), dim=-1)

#         for i in range(len(self.fc_layers)):
#             h = F.relu(self.fc_layers[i](h))

#         p_x = self.fc_out(h)
#         if self.pred_type == 'deterministic' or self.pred_type == 'gaussian':
#             pass
#         elif self.pred_type == 'bernoulli':
#             p_x = torch.sigmoid(p_x)
#         elif self.pred_type == 'categorical':
#             p_x = torch.softmax(p_x, 1)
#         else:
#             raise NotImplementedError

#         return p_x


class TaskDecoder(nn.Module):
    def __init__(self,
                 layers,
                 task_embedding_size,
                 pred_type,
                 task_dim,
                 ):
        super(TaskDecoder, self).__init__()

        # "task_description" or "task id"
        self.pred_type = pred_type

        curr_input_size = task_embedding_size
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
            curr_input_size = layers[i]

        self.fc_out = nn.Linear(curr_input_size, task_dim)

    def forward(self, task_embedding):

        h = task_embedding

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        y = self.fc_out(h)

        if self.pred_type == 'task_id':
            y = torch.softmax(y, 1)

        return y