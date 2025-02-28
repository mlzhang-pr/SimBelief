import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu


class RNNEncoder(nn.Module):
    def __init__(self,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 task_embedding_size=32,
                 # actions, states, rewards
                 action_size=2,   # 在主文件中初始化过
                 action_embed_size=10,
                 state_size=2,
                 state_embed_size=10,
                 reward_size=1,
                 reward_embed_size=5,
                 #
                 distribution='gaussian',
                 ):
        super(RNNEncoder, self).__init__()

        self.task_embedding_size = task_embedding_size
        self.hidden_size = hidden_size
        self.gru_h=None
        if distribution == 'gaussian':
            self.reparameterise = self._sample_gaussian
        else:
            raise NotImplementedError

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)

        # fully connected layers before the recurrent cell
        curr_input_size = action_embed_size + state_embed_size + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_size, layers_before_gru[i]))
            curr_input_size = layers_before_gru[i]

        # recurrent unit
        self.gru = nn.GRU(input_size=curr_input_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_size = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_size, layers_after_gru[i]))
            curr_input_size = layers_after_gru[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_size, task_embedding_size)
        self.fc_logvar = nn.Linear(curr_input_size, task_embedding_size)

    def _sample_gaussian(self, mu, logvar, num=None): 
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            if logvar.shape[0] > 1:
                mu = mu.unsqueeze(0)
                logvar = logvar.unsqueeze(0)
            if logvar.dim() > 2:    # if 3 dims, first must be 1
                assert logvar.shape[0] == 1, 'error in dimensions!'
                std = torch.exp(0.5 * logvar).repeat(num, 1, 1)
                eps = torch.randn_like(std)
                mu = mu.repeat(num, 1, 1)
            else:
                std = torch.exp(0.5 * logvar).repeat(num, 1)
                eps = torch.randn_like(std)
                mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, reset_task):
        if hidden_state.dim() != reset_task.dim():
            if reset_task.dim() == 2:
                reset_task = reset_task.unsqueeze(0)
            elif reset_task.dim() == 1:
                reset_task = reset_task.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - reset_task)
        return hidden_state

    def prior(self, batch_size, sample=True):

        # TODO: somehow incorporate the initial state

        # we start out with a hidden state of zero
        # hidden_state = torchkit.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(ptu.device)
        hidden_state = ptu.zeros((1, batch_size, self.hidden_size), requires_grad=True)

        h = hidden_state
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        task_mean = self.fc_mu(h)
        task_logvar = self.fc_logvar(h)
        if sample:
            task_sample = self.reparameterise(task_mean, task_logvar) 
        else:
            task_sample = task_mean

        return task_sample, task_mean, task_logvar, hidden_state

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True):
        """
        Actions, states, rewards should be given in form [sequence_len * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """
        # shape should be: sequence_len x batch_size x hidden_size
        if actions.dim() != 3:
            actions = actions.unsqueeze(dim=1)  # 增加一个batchsize维度
            states = states.unsqueeze(dim=1)
            rewards = rewards.unsqueeze(dim=1)

        if hidden_state is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))
            # hidden_state = hidden_state.unsqueeze(dim=1)

        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(actions.shape[1])
            hidden_state = prior_hidden_state.clone()

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=-1)

        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        # GRU cell (output is outputs for each time step, hidden_state is last output)
        output, _ = self.gru(h, hidden_state)
        # gru_h = F.relu(output)  # TODO: should this be here?
        gru_h = output.clone()
        self.gru_h=gru_h

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        # outputs
        task_mean = self.fc_mu(gru_h)
        task_logvar = self.fc_logvar(gru_h)
        if sample:
            task_sample = self.reparameterise(task_mean, task_logvar)
        else:
            task_sample = task_mean

        if return_prior:
            task_sample = torch.cat((prior_sample, task_sample))
            task_mean = torch.cat((prior_mean, task_mean))
            task_logvar = torch.cat((prior_logvar, task_logvar))
            output = torch.cat((prior_hidden_state, output))    # (61, 16, 64)
            gru_h=output

        if task_mean.shape[0] == 1:
            task_sample, task_mean, task_logvar = task_sample[0], task_mean[0], task_logvar[0]

        return task_sample, task_mean, task_logvar, output, gru_h

class CorrelationNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CorrelationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 使用Tanh将输出范围约束在[-1, 1]之间
        )

    def forward(self, rnn_output):
        return self.fc(rnn_output)

class WeightNetwork(nn.Module):
    def __init__(self, input_dim):
        super(WeightNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)  # 输出两个权重

    def forward(self, rnn_output):
        x = torch.relu(self.fc1(rnn_output))
        weights = torch.softmax(self.fc2(x), dim=-1)  # 使用softmax将权重归一化
        return weights


class TaskIdBisimEncoder(nn.Module): # training with SAC
    def __init__(self,
                 # network size
                task_embedding_size=10,
                hidden_dim=64,   # point robot 256  control task 64
                rnn_output_size=128,
                ):
        super(TaskIdBisimEncoder, self).__init__()
        
        self.fc1 = nn.Linear(rnn_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, task_embedding_size)
        self.logvar_head = nn.Linear(hidden_dim, task_embedding_size)
        self.max_norm=20
    def forward(self, rnn_output,normalize=False):
        
        h = rnn_output
        # h=F.leaky_relu(rnn_output)
        # h=h.unsqueeze(dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))

        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        
        
        if self.max_norm and normalize:
            mean, logvar = self.normalize(mean, logvar)
        logvar = torch.clamp(logvar, min=-10, max=4)

        return mean, logvar
    
    def normalize(self,x,y):
        epsilon = 1e-8
        if self.max_norm:
            norms = x.norm(dim=-1) + epsilon
            norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
            x = x / norm_to_max
            y = y / norm_to_max
        return x, y

class AdaptiveEncoder(nn.Module): # training with SAC 
    def __init__(self,
                 # network size
                task_embedding_size=10,
                input_dim=20,   # point robot 256  control task 64
                
                ):
        super(AdaptiveEncoder, self).__init__()
        
        self.mean_head = nn.Linear(input_dim, task_embedding_size)
        self.logvar_head = nn.Linear(input_dim, task_embedding_size)

    def forward(self, final_mean, final_logvar):
        
        h=torch.cat((final_mean, final_logvar),dim=-1)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        
        return mean, logvar



class BeliefCombineEncoder(nn.Module): # training with VAE
    def __init__(self,
                 # network size
                task_embedding_size=10,
                hidden_dim=64,   # point robot 256  control task 64
                input_size=40,
                ):
        super(BeliefCombineEncoder, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, task_embedding_size)
        self.logvar_head = nn.Linear(hidden_dim, task_embedding_size)

    def forward(self, curr_means, curr_logvars, mean_shift, logvar_shift):
        
        h = torch.cat((curr_means, curr_logvars, mean_shift,logvar_shift),dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        
        return mean, logvar
    
    
class TaskIdBisimEncoderRNN(nn.Module): # training with SAC
    def __init__(self,
                 # network size
                task_embedding_size=32,
                hidden_dim=64,   # point robot 256  control task 64
                rnn_output_size=64,
                ):
        super(TaskIdBisimEncoderRNN, self).__init__()
        
        self.gru = nn.GRU(input_size=rnn_output_size,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          )

        # self.fc1 = nn.Linear(rnn_output_size, hidden_dim)
        
        self.mean_head = nn.Linear(hidden_dim, task_embedding_size)
        self.logvar_head = nn.Linear(hidden_dim, task_embedding_size)

    def forward(self, rnn_output):
        

        h = rnn_output
        h, _ = self.gru(h)
        # h = h[:, -1, :]

        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        
        return mean, logvar

class DistShiftEncoder(nn.Module):
    def __init__(self,
                 # network size
                task_embedding_size=10,
                hidden_dim=64,   # point robot 256  control task 64
                rnn_output_size=128,
                ):
        super(DistShiftEncoder, self).__init__()
        
        self.fc1 = nn.Linear(rnn_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, task_embedding_size)
        self.logvar_head = nn.Linear(hidden_dim, task_embedding_size)
        self.max_norm=20
    def forward(self, rnn_output, normalize=False):
        
        h = rnn_output
        # h=F.leaky_relu(rnn_output)
        # h=h.unsqueeze(dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))

        mean = self.mean_head(h)
        var = self.logvar_head(h)
        
        
        if self.max_norm and normalize:
            mean, logvar = self.normalize(mean, logvar)
        # logvar = torch.clamp(logvar, min=-10, max=4)
        var = torch.clamp(var, min=1e-6)

        return mean, var
    
    def normalize(self,x,y):
        epsilon = 1e-8
        if self.max_norm:
            norms = x.norm(dim=-1) + epsilon
            norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
            x = x / norm_to_max
            y = y / norm_to_max
        return x, y




# class DistShiftEncoder(nn.Module):
#     def __init__(self,
#                  # network size
#                 task_embedding_size=32,
#                 hidden_dim=64,   # point robot 256  control task 64
#                 # actions, states, rewards
#                 action_size=2,
#                 action_embed_size=10,
#                 state_size=50,
#                 state_embed_size=64,
#                 reward_size=1,
#                 reward_embed_size=5,
#                 next_state_size=2,
#                 next_state_embed_size=10,

#                 ):
#         super(DistShiftEncoder, self).__init__()

#         # embed action, state
#         self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
#         self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)
#         self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)
        
#         self.fc1 = nn.Linear(state_embed_size + action_embed_size +reward_embed_size, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)

#         self.mean_head = nn.Linear(hidden_dim, task_embedding_size)
#         self.logvar_head = nn.Linear(hidden_dim, task_embedding_size)

#     def forward(self, state, action, reward):
#         if action.dim() != 3:
#             action = action.unsqueeze(dim=1)  # 增加一个batchsize维度, 在update调用时需要reshape
#             state = state.unsqueeze(dim=1)
#             reward = reward.unsqueeze(dim=1)
#         hs = self.state_encoder(state)
#         ha= self.action_encoder(action)
#         hr = self.reward_encoder(reward)
 
#         h = torch.cat((hs,ha,hr), dim=-1)
        
#         h = F.relu(self.fc1(h))
#         h = F.relu(self.fc2(h))

#         mean = self.mean_head(h)
#         logvar = self.logvar_head(h)
        
#         return mean, logvar



# class DistShiftEncoder(nn.Module):
#     def __init__(self,
#                  # network size
#                 task_embedding_size=32,
#                 hidden_dim=128,
#                 # actions, states, rewards
#                 action_size=2,
#                 action_embed_size=10,
#                 state_size=2,
#                 state_embed_size=10,
#                 reward_size=1,
#                 reward_embed_size=5,
#                 next_state_size=2,
#                 next_state_embed_size=10,

#                 ):
#         super(DistShiftEncoder, self).__init__()

#         # embed action, state
#         self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
#         self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)
#         self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)
        
#         self.fc1 = nn.Linear(state_embed_size + action_embed_size + state_embed_size + reward_embed_size, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)

#         self.mean_head = nn.Linear(hidden_dim, task_embedding_size)
#         self.logvar_head = nn.Linear(hidden_dim, task_embedding_size)

#     def forward(self, state, action, next_state, reward):
#         ha = self.action_encoder(action)
#         hs = self.state_encoder(state)
#         hr = self.reward_encoder(reward)
#         hs_=self.state_encoder(next_state)
#         h = torch.cat((hs,ha,hs_,hr), dim=-1)
        
#         h = F.relu(self.fc1(h))
#         h = F.relu(self.fc2(h))

#         mean = self.mean_head(h)
#         logvar = self.logvar_head(h)
        
#         return mean, logvar



class StateActionEncoder(nn.Module):
    def __init__(self,
                 # network size
                 
                 layers_after_gru=(),
                 task_embedding_size=32,
                 hidden_dim=64,
                 # actions, states, rewards
                 action_size=2,
                 action_embed_size=10,
                 state_size=2,
                 state_embed_size=10,
                 reward_size=1,
                 reward_embed_size=5,
                 #
                 distribution='gaussian',
                 ):
        super(StateActionEncoder, self).__init__()

        # embed action, state
        self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_size, action_embed_size, F.relu)
        
        self.fc1 = nn.Linear(state_embed_size + action_embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_head = nn.Linear(hidden_dim, task_embedding_size)
        self.logvar_head = nn.Linear(hidden_dim, task_embedding_size)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        
        return mean, logvar

