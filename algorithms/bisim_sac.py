"""
Based on https://github.com/pranz24/pytorch-soft-actor-critic
"""

import os
import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim
import torchkit.pytorch_utils as ptu
from models.bisimencoder import make_encoder
from models.transition_model import make_transition_model
from models.encoder import (
    DistShiftEncoder,
    TaskIdBisimEncoder,
    TaskIdBisimEncoderRNN,
    CorrelationNetwork,
    WeightNetwork,
    BeliefCombineEncoder,
    AdaptiveEncoder,
)


class BisimSAC(nn.Module):
    def __init__(
        self,
        policy,
        q1_network,
        q2_network,
        vae,
        augmented_obs_dim,
        action_dim,
        action_embed_size,
        obs_dim,
        state_embed_size,
        reward_size,
        reward_embed_size,
        c_R,
        c_T,
        z_dim,
        encoder_type="statebelief",  # TODO: put in args
        encoder_feature_dim=50,
        encoder_layers=[64, 64],  # point robot [256,128], continous control [64,64]
        # encoder_layers=[64,64,64],
        use_indyn=True,
        transition_model_type="probabilistic",
        encoder_lr=3e-4,
        decoder_lr=3e-4,
        bisim_coef=0.5,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        #  vae_loss=None,
        use_cql=False,
        alpha_cql=2.0,
        entropy_alpha=0.2,
        automatic_entropy_tuning=True,
        alpha_lr=3e-4,
        clip_grad_value=18.0,
    ):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.use_cql = use_cql  # Conservative Q-Learning loss
        self.alpha_cql = alpha_cql  # Conservative Q-Learning weight parameter
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.clip_grad_value = clip_grad_value
        self.bisim_coef = bisim_coef
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.use_indyn = use_indyn
        self.c_R = c_R
        self.c_T = c_T
        self.pass_to_distshfit = False
        # self.vae_loss=vae_loss
        self.use_mlp_rho = False
        self.compute_weight = False
        self.update_latent_belief = False
        self.combine_outside=True
        encoder_max_norm = 150

        # belief bisimulation
        self.transition_model = vae.state_decoder  # for one step prediction
        self.reward_model = vae.reward_decoder
        self.state_encoder = make_encoder(
            "state_encoder",
            obs_dim,
            encoder_feature_dim,
            max_norm=None,
            layers=encoder_layers,
        )
        self.context_encoder = vae.encoder
        self.vae = vae
        # self.distshift_encoder = DistShiftEncoder(task_embedding_size=z_dim,
        #                                         action_size=action_dim,
        #                                         action_embed_size=action_embed_size,
        #                                         state_size=obs_dim,
        #                                         state_embed_size=state_embed_size,
        #                                         reward_embed_size=reward_embed_size)
        self.distshift_encoder = DistShiftEncoder(
            task_embedding_size=z_dim,
            hidden_dim=64,
            rnn_output_size=128,
        )

        # self.task_encoder = TaskIdBisimEncoderRNN(task_embedding_size=z_dim,
        #                                        hidden_dim=16,
        #                                         rnn_output_size=128,)
        self.task_encoder = TaskIdBisimEncoder(
            task_embedding_size=z_dim,
            hidden_dim=64,
            rnn_output_size=128,
        )

        self.adaptive_encoder = AdaptiveEncoder(task_embedding_size=z_dim, input_dim=20)
        # self.bisim_belief_encoder = BeliefCombineEncoder(task_embedding_size=z_dim,
        #                                                 hidden_dim=64,
        #                                                 input_size=z_dim*4,).to(ptu.device)

        if self.use_mlp_rho:
            self.rho_encoder = CorrelationNetwork(128)
            self.rho_optimizer = torch.optim.Adam(
                self.rho_encoder.parameters(), lr=encoder_lr
            )

        if self.compute_weight:
            self.w1 = 0.5
            self.w2 = 0.5
            # self.w1=nn.Parameter(torch.sigmoid(torch.tensor(0.5)))
            # self.w2=nn.Parameter(torch.sigmoid(torch.tensor(0.5)))
            # self.gmm_w_optimizer = torch.optim.Adam([self.w1, self.w2], lr=encoder_lr)
            self.weight_network = WeightNetwork(128)
            self.weight_network_optimizer = torch.optim.Adam(
                self.weight_network.parameters(), lr=encoder_lr
            )

        # self.critic_encoder=make_encoder(encoder_type,augmented_obs_dim,encoder_feature_dim,max_norm=encoder_max_norm,layers=encoder_layers)
        self.critic = Critic(q1_network, q2_network)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.next_distshift_encoder = copy.deepcopy(self.distshift_encoder)
        
        
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.encoder_optimizer = torch.optim.Adam(
            self.state_encoder.parameters(), lr=encoder_lr
        )
        self.distshift_encoder_optimizer = torch.optim.Adam(
            self.distshift_encoder.parameters(), lr=encoder_lr
        )

        self.task_encoder_optimizer = torch.optim.Adam(
            self.task_encoder.parameters(), lr=encoder_lr
        )

        self.adaptive_encoder_optimizer = torch.optim.Adam(
            self.adaptive_encoder.parameters(), lr=encoder_lr
        )

        # self.bisim_belief_encoder_optimizer = torch.optim.Adam(
        #     self.bisim_belief_encoder.parameters(), lr=encoder_lr
        # )

        # invdyn_model
        if use_indyn:
            self.setup_inverse_dynamic_model(
                z_dim + encoder_feature_dim, action_dim, invdyn_lr=3e-4
            )

        # q networks - use two network to mitigate positive bias
        self.qf1 = self.critic.Q1
        self.qf2 = self.critic.Q2
        self.qf1_target = self.critic_target.Q1
        self.qf2_target = self.critic_target.Q2

        self.policy = policy
        # self.policy.encoder.copy_conv_weights_from(self.critic.encoder)  #################
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)  #  weight_decy

        # automatic entropy coefficient tuning
        if self.automatic_entropy_tuning:
            # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(ptu.device)).item()
            self.target_entropy = -self.policy.action_dim
            self.log_alpha_entropy = torch.zeros(
                1, requires_grad=True, device=ptu.device
            )
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp()
        else:
            self.alpha_entropy = entropy_alpha

    def forward(self, obs):
        action, _, _, _ = self.policy(obs)
        q1, q2 = self.qf1(obs, action), self.qf2(obs, action)
        return action, q1, q2

    def act(self, obs, deterministic=False, return_log_prob=False):
        action, mean, log_std, log_prob = self.policy(
            obs, deterministic=deterministic, return_log_prob=return_log_prob
        )
        return action, mean, log_std, log_prob

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(ptu.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def setup_inverse_dynamic_model(
        self,
        augmented_encode_state_dim,
        action_shape,
        hidden_layers=(64, 64),
        invdyn_lr=3e-4,
    ):  # 64，64
        self.invdyn_model = nn.Sequential(
            nn.Linear(2 * augmented_encode_state_dim, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ELU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ELU(),
            nn.Linear(hidden_layers[1], action_shape),
            nn.Tanh(),
        )
        self.invdyn_optimizer = torch.optim.Adam(
            self.invdyn_model.parameters(), lr=invdyn_lr, weight_decay=1e-5
        )  # weight_decay = 1e-5

    def update_context_encoder(
        self, augmented_obs, action, augmented_next_obs, reward, t_dim, b_dim
    ):  # belief bisimulation loss
        belief = augmented_obs[
            :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 4
        ]  # get current belief (t*b,_)
        belief_next = augmented_next_obs[
            :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 4
        ]

        state = augmented_obs[:, : self.obs_dim]
        next_state = augmented_next_obs[:, : self.obs_dim]
        belief_mu = belief[:, : belief.size(-1) // 2]
        belief_logvar = belief[:, belief.size(-1) // 2 :]
        # belief_next_mu=belief_next[:,:belief_next.size(-1)//2]
        # belief_next_logvar=belief_next[:,belief_next.size(-1)//2:]
        belief_next_mu = belief_next[:, : belief_next.size(-1) // 2]
        belief_next_logvar = belief_next[:, belief_next.size(-1) // 2 :]

        z1 = self.context_encoder._sample_gaussian(
            belief_mu, belief_logvar
        )  # sample z from belief ,test: directly get current z
        z1_raw = z1.view(t_dim, b_dim, -1)
        z2_raw = torch.flip(z1_raw, [0])
        z2 = z2_raw.view(t_dim * b_dim, -1)

        z1_next = self.context_encoder._sample_gaussian(
            belief_next_mu, belief_next_logvar
        )
        z1_next_raw = z1_next.view(t_dim, b_dim, -1)
        z2_next_raw = torch.flip(z1_next_raw, [0])
        z2_next = z2_next_raw.view(t_dim * b_dim, -1)

    
        b1 = belief
        b2 = torch.flip(b1, [0])

        # with torch.no_grad():
        encode_state = self.state_encoder(state)
        encode_nextstate = self.state_encoder(next_state)

        augmented_state1 = torch.cat([encode_state, z1], dim=-1)
        augmented_state2 = torch.cat([encode_state, z2], dim=-1)

        augmented_next_state1 = torch.cat([encode_nextstate, z1_next], dim=-1)
        augmented_next_state2 = torch.cat([encode_nextstate, z2_next], dim=-1)
        with torch.no_grad():
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(
                augmented_state1, action
            )
            pred_next_latent_mu2, pred_next_latent_sigma2 = self.transition_model(
                augmented_state2, action
            )

            reward1 = self.reward_model(
                augmented_next_state1, None, action
            )  # deterministic
            reward2 = self.reward_model(augmented_next_state2, None, action)

        z_dist = F.smooth_l1_loss(z1, z2, reduction="none")
        r_dist = F.smooth_l1_loss(reward1, reward2, reduction="none")

        # if sparse point robot
        # transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
        # else:
        # transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none') \
        #             + F.smooth_l1_loss(pred_next_latent_sigma1, pred_next_latent_sigma2, reduction='none')  ###########修改过
        transition_dist = torch.sqrt(
            (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2)
            + (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
        )

        # transition_dist = torch.norm(pred_next_latent_mu1 - pred_next_latent_mu2) + torch.norm(pred_next_latent_sigma1 - pred_next_latent_sigma2)

        if self.use_indyn:
            with torch.no_grad():
                action1 = self.invdyn_model(
                    torch.cat([augmented_state1, augmented_next_state1], dim=-1)
                )
                action2 = self.invdyn_model(
                    torch.cat([augmented_state2, augmented_next_state2], dim=-1)
                )
            epsilon = 1e-8
            action1_normalized = (action1 - action1.mean()) / action1.std()
            action2_normalized = (action2 - action2.mean()) / action2.std()
            action_dist = (action1.float() - action2.float()).abs().mean()
            # action_dist=torch.norm(action1.float() - action2.float())
            # action_dist=(action1_normalized.float() - action2_normalized.float()).abs().mean()
            # action_dist=F.smooth_l1_loss(action1.float(), action2.float(), reduction='none')

            bisimilarity = r_dist + transition_dist + action_dist  # without scale
        else:
            bisimilarity = r_dist + transition_dist

        beliefbisim_loss = F.mse_loss(torch.norm(z_dist), torch.norm(bisimilarity))

        return beliefbisim_loss, z1, z1_next

    def overlap_dist(self, mean1, mean2, logvar1, logvar2, normal=True):

        sigma1 = torch.exp(0.5 * logvar1)
        sigma2 = torch.exp(0.5 * logvar2)

        w1, w2 = 0.5, 0.5
        rho = 0
        if normal:
            mean_new = w1 * mean1 + w2 * mean2
            var_new = (
                w1**2 * sigma1**2
                + w2**2 * sigma2**2
                + 2 * w1 * w2 * rho * sigma1 * sigma2
            )
            logvar_new = torch.log(var_new)


        return mean_new, logvar_new

    def compute_rho(self, mean1, logvar1, mean2, logvar2, rnn_output):
        if self.use_mlp_rho:
            rnn_output = rnn_output
            rho = self.rho_encoder(rnn_output)
            rho = torch.sigmoid(rho) * 2 - 1
        else:
            sigma1 = torch.exp(0.5 * logvar1)
            sigma2 = torch.exp(0.5 * logvar2)
    
            z1 = torch.normal(mean1, sigma1)
            z2 = torch.normal(mean2, sigma2)

            mean_z1 = torch.mean(z1)
            mean_z2 = torch.mean(z2)

            cov_z1_z2 = torch.mean((z1 - mean_z1) * (z2 - mean_z2))

            sigma1_sample = torch.sqrt(torch.mean((z1 - mean_z1) ** 2))
            sigma2_sample = torch.sqrt(torch.mean((z2 - mean_z2) ** 2))

            rho = cov_z1_z2 / (sigma1_sample * sigma2_sample)
        return rho

    def combine_gaussian_distributions(
        self, mean1, logvar1, mean2, logvar2, rnn_output, iter_, w1=0.5, w2=0.5
    ):  

        if self.compute_weight == True:
            if iter_ == 0:
                w1, w2 = 0.5, 0.5
            else:
                weights = self.weight_network(rnn_output)
                w1, w2 = weights.chunk(2, dim=-1)

        non_corr = True
        if non_corr == True:
            mean_new = w1 * mean1 + w2 * mean2

            var1 = torch.exp(logvar1)
            var2 = torch.exp(logvar2)
            var_mix = w1 * (var1 + mean1**2) + w2 * (var2 + mean2**2) - mean_new**2
            var_mix = torch.clamp(var_mix, min=1e-6)
            logvar_new = torch.log(var_mix)
        else:
            # 计算标准差
            sigma1 = torch.exp(0.5 * logvar1)
            sigma2 = torch.exp(0.5 * logvar2)

            rho = self.compute_rho(mean1, logvar1, mean2, logvar2, rnn_output)

            mean_new = w1 * mean1 + w2 * mean2
            var_new = (
                w1**2 * sigma1**2
                + w2**2 * sigma2**2
                + 2 * w1 * w2 * rho * sigma1 * sigma2
            )

            logvar_new = torch.log(var_new)

        return mean_new, logvar_new

    def combine_gaussian_distributions_eval(
        self, mean1, logvar1, mean2, logvar2, rnn_output, w1=0.5, w2=0.5
    ):  
        if self.compute_weight == True:
            with torch.no_grad():
                weights = self.weight_network(rnn_output)
            w1, w2 = weights.chunk(2, dim=-1)

        non_corr = True
        if non_corr == True:
            mean_new = w1 * mean1 + w2 * mean2

            var1 = torch.exp(logvar1)
            var2 = torch.exp(logvar2)
            var_mix = w1 * (var1 + mean1**2) + w2 * (var2 + mean2**2) - mean_new**2
            var_mix = torch.clamp(var_mix, min=1e-6)
            logvar_new = torch.log(var_mix)
        else:
            sigma1 = torch.exp(0.5 * logvar1)
            sigma2 = torch.exp(0.5 * logvar2)

            rho = self.compute_rho(mean1, logvar1, mean2, logvar2, rnn_output)

            mean_new = w1 * mean1 + w2 * mean2
            var_new = (
                w1**2 * sigma1**2
                + w2**2 * sigma2**2
                + 2 * w1 * w2 * rho * sigma1 * sigma2
            )

            logvar_new = torch.log(var_new)

        return mean_new, logvar_new

    def compute_mutual_information(
        self, mean1, logvar1, mean2, logvar2, rnn_output=None
    ):
        rho = self.compute_rho(mean1, logvar1, mean2, logvar2, rnn_output)
        mutual_information = -0.5 * torch.log(1 - rho**2)
        return torch.mean(mutual_information)

    def update_context_encoder_cur(
        self,
        augmented_obs,
        action,
        augmented_next_obs,
        reward,
        t_dim,
        b_dim,
        use_tencoder=True,
    ):  # belief bisimulation loss

        state = augmented_obs[:, : self.obs_dim]
        next_state = augmented_next_obs[:, : self.obs_dim]

        add_vae_belief = False
        use_curr_rnn = False

        if use_tencoder:
            gru_h = augmented_obs[:, self.obs_dim + self.z_dim * 4 :]
            next_gru_h = augmented_next_obs[:, self.obs_dim + self.z_dim * 4 :]
            if use_curr_rnn:
                _, latent_mean, latent_logvar, _, rnn_output = self.vae.encoder(
                    actions=action,
                    states=next_state,
                    rewards=reward,
                    hidden_state=None,
                )
                belief_mu, belief_logvar = self.task_encoder(rnn_output)
                belief_next_mu, belief_next_logvar = self.task_encoder(next_gru_h)

            else:
                belief_mu, belief_logvar = self.task_encoder(gru_h)
                belief_next_mu, belief_next_logvar = self.task_encoder(next_gru_h)

            if add_vae_belief:
                vae_belief = augmented_obs[
                    :, self.obs_dim : self.obs_dim + self.z_dim * 2
                ]  # get current belief (t*b,_)
                vae_belief_next = augmented_next_obs[
                    :, self.obs_dim : self.obs_dim + self.z_dim * 2
                ]
                vae_belief_mu = vae_belief[:, : vae_belief.size(-1) // 2]
                vae_belief_logvar = vae_belief[:, vae_belief.size(-1) // 2 :]
                vae_belief_next_mu = vae_belief_next[:, : vae_belief.size(-1) // 2]
                vae_belief_next_logvar = vae_belief_next[:, vae_belief.size(-1) // 2 :]

                # belief_mu,belief_logvar=self.combine_gaussian_distributions(belief_mu,belief_logvar,vae_belief_mu,vae_belief_logvar,gru_h,None)
                # belief_next_mu,belief_next_logvar=self.combine_gaussian_distributions(belief_next_mu,belief_next_logvar,vae_belief_next_mu,vae_belief_next_logvar,next_gru_h,None)
                with torch.no_grad():
                    belief_mu, belief_logvar = self.vae.belief_encoder(
                        vae_belief_mu, vae_belief_logvar, belief_mu, belief_logvar
                    )
                    belief_next_mu, belief_next_logvar = self.vae.belief_encoder(
                        vae_belief_next_mu,
                        vae_belief_next_logvar,
                        belief_next_mu,
                        belief_next_logvar,
                    )

        else:
            belief = augmented_obs[
                :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 4
            ]  # get current belief (t*b,_)
            belief_next = augmented_next_obs[
                :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 4
            ]

            belief_mu = belief[:, : belief.size(-1) // 2]
            belief_logvar = belief[:, belief.size(-1) // 2 :]

            belief_next_mu = belief_next[:, : belief_next.size(-1) // 2]
            belief_next_logvar = belief_next[:, belief_next.size(-1) // 2 :]

        perm = np.random.permutation(t_dim)
        z1 = self.context_encoder._sample_gaussian(
            belief_mu, belief_logvar
        )  # sample z from belief ,test: directly get current z
        z1_raw = z1.view(t_dim, b_dim, -1)
        # z2_raw=torch.flip(z1_raw,[0])
        # z2_raw=z1_raw[perm]
        z2_raw = torch.flip(z1_raw, [0])
        z2 = z2_raw.view(t_dim * b_dim, -1)

        z1_next = self.context_encoder._sample_gaussian(
            belief_next_mu, belief_next_logvar
        )
        z1_next_raw = z1_next.view(t_dim, b_dim, -1)
        z2_next_raw = torch.flip(z1_next_raw, [0])
        z2_next = z2_next_raw.view(t_dim * b_dim, -1)

        # b1=belief
        # b2=torch.flip(b1,[0])

        # with torch.no_grad(): # ?
        encode_state = self.state_encoder(state)
        encode_nextstate = self.state_encoder(next_state)

        augmented_state1 = torch.cat([encode_state, z1], dim=-1)
        augmented_state2 = torch.cat([encode_state, z2], dim=-1)

        augmented_next_state1 = torch.cat([encode_nextstate, z1_next], dim=-1)
        augmented_next_state2 = torch.cat([encode_nextstate, z2_next], dim=-1)
        with torch.no_grad():  #######################
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(
                augmented_state1, action
            )
            pred_next_latent_mu2, pred_next_latent_sigma2 = self.transition_model(
                augmented_state2, action
            )

            reward1 = self.reward_model(
                augmented_next_state1, None, action
            )  # deterministic
            reward2 = self.reward_model(augmented_next_state2, None, action)
            # reward2 = reward.view(t_dim,b_dim,-1)
            # reward2 = reward2[perm]
            # reward2 = reward2.view(t_dim*b_dim,-1)

        z_dist = F.smooth_l1_loss(z1, z2, reduction="none")
        r_dist = F.smooth_l1_loss(reward1, reward2, reduction="none")

        # transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none') \
        #             + F.smooth_l1_loss(pred_next_latent_sigma1, pred_next_latent_sigma2, reduction='none')
        # transition_dist = torch.sqrt(
        #         (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
        #         (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
        #     ).mean(dim=1)
        transition_dist = torch.sqrt(
            (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2)
            + (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
        )

        if self.use_indyn:
            with torch.no_grad():  ###################
                action1 = self.invdyn_model(
                    torch.cat([augmented_state1, augmented_next_state1], dim=-1)
                )
                action2 = self.invdyn_model(
                    torch.cat([augmented_state2, augmented_next_state2], dim=-1)
                )
            epsilon = 1e-8
            action1_normalized = (action1 - action1.mean()) / action1.std()
            action2_normalized = (action2 - action2.mean()) / action2.std()
            # action_dist=F.smooth_l1_loss(action1.float(), action2.float(), reduction='none').mean()
            # action_dist=(action1.float() - action2.float()).abs().mean(dim=1)
            action_dist = (action1.float() - action2.float()).abs().mean()

            bisimilarity = r_dist + transition_dist + action_dist  # without scale
        else:
            bisimilarity = r_dist + transition_dist

        beliefbisim_loss = F.mse_loss(torch.norm(z_dist), torch.norm(bisimilarity))
        return beliefbisim_loss, z1, z1_next

    def update_transition_reward_decoder(
        self, augmented_obs, action, augmented_next_obs, reward
    ):  # next step reconstruction
        belief = augmented_obs[
            :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 4
        ]  # get current belief (t*b,_)
        belief_next = augmented_next_obs[
            :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 4
        ]
        state = augmented_obs[:, : self.obs_dim]
        shift_next = augmented_next_obs[:, self.obs_dim + self.z_dim * 2 :]
        next_state = augmented_next_obs[:, : self.obs_dim]
        belief_mu = belief[:, : belief.size(1) // 2]
        belief_logvar = belief[:, belief.size(1) // 2 :]
        belief_next_mu = belief_next[:, : belief_next.size(1) // 2]
        belief_next_logvar = belief_next[:, belief_next.size(1) // 2 :]

        shift = augmented_obs[:, self.obs_dim + self.z_dim * 2 :]
        # TODO: compress belief with shift
        shift_mu = shift[:, : shift.size(-1) // 2]
        shift_logvar = shift[:, shift.size(-1) // 2 :]

        if self.pass_to_distshfit:
            shift_next_mu, shift_next_logvar = self.distshift_encoder(
                next_state, action, reward
            )
            t, b, _ = shift_next_mu.size()
            shift_next_mu = shift_next_mu.view(t * b, -1)
            shift_next_logvar = shift_next_logvar.view(t * b, -1)
        else:
            shift_next_mu = shift_next[:, : shift_next.size(-1) // 2]
            shift_next_logvar = shift_next[:, shift_next.size(-1) // 2 :]

        # -------- use dist shift ------
        # belief_mu, belief_logvar=self.compress_dist(belief_mu, belief_logvar,shift_mu,shift_logvar)  # compress belief with shift
        # belief_next_mu, belief_next_logvar=self.compress_dist(belief_next_mu,belief_next_logvar,shift_next_mu,shift_next_logvar)

        z1 = self._sample_gaussian(
            belief_mu, belief_logvar
        )  # sample z from belief ,test: directly get current z
        z1_next = self._sample_gaussian(belief_next_mu, belief_next_logvar)

        encode_state = self.state_encoder(state)
        encode_nextstate = self.state_encoder(next_state)
        augmented_state1 = torch.cat([encode_state, z1], dim=1)
        augmented_next_state1 = torch.cat([encode_nextstate, z1_next], dim=1)
        # sparse point robot  ignore next step state
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(
            augmented_state1, action
        )
        encode_nextstate_detach = encode_nextstate.detach()
        diff = (
            pred_next_latent_mu - encode_nextstate_detach
        ) / pred_next_latent_sigma  # detach?
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))

        pred_reward = self.reward_model(augmented_next_state1, None, action)
        # reward_loss=F.mse_loss(pred_reward,reward, reduction='none')
        reward_loss = F.mse_loss(pred_reward, reward)

        if self.use_indyn:
            pred_action = self.invdyn_model(
                torch.cat([augmented_state1, augmented_next_state1], dim=-1)
            )
            action_loss = (action - pred_action).abs().mean()
            total_loss = loss + reward_loss + action_loss
        else:
            total_loss = loss + reward_loss
        # total_loss=reward_loss  # only reward loss
        return total_loss

    def compute_kl_loss(
        self, augmented_obs, augmented_next_obs, kl_to_gauss_prior=False
    ):
        gru_h = augmented_obs[:, self.obs_dim + self.z_dim * 4 :]
        next_gru_h = augmented_next_obs[:, self.obs_dim + self.z_dim * 4 :]
        belief_mu, belief_logvar = self.task_encoder(gru_h)
        belief_next_mu, belief_next_logvar = self.task_encoder(next_gru_h)

        # -- KL divergence
        if kl_to_gauss_prior:
            kl_divergences = -0.5 * (
                1 + belief_logvar - belief_mu.pow(2) - belief_logvar.exp()
            ).sum(dim=1)
        else:
            gauss_dim = belief_mu.shape[-1]
            # add the gaussian prior
            # https://arxiv.org/pdf/1811.09975.pdf
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
            mu = belief_next_mu
            m = belief_mu
            logE = belief_next_logvar
            logS = belief_logvar
            kl_divergences = 0.5 * (
                torch.sum(logS, dim=1)
                - torch.sum(logE, dim=1)
                - gauss_dim
                + torch.sum(1 / torch.exp(logS) * torch.exp(logE), dim=1)
                + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=1)
            )

        return kl_divergences.mean()

    def kl_b1_b2(self, mean_b1, logvar_b1, mean_b2, logvar_b2):
        var = torch.exp(logvar_b2)
        var_target = torch.exp(logvar_b1)

        kl_div = 0.5 * (
            (var / var_target)
            + ((mean_b1 - mean_b2) ** 2) / var_target
            - 1
            + torch.log(var_target / var)
        )
        return kl_div.mean()

    def update_transition_reward_decoder_cur(
        self, augmented_obs, action, augmented_next_obs, reward, use_tencoder=True
    ):  # next step reconstruction
        # try use_tencoder=False
        state = augmented_obs[:, : self.obs_dim]
        next_state = augmented_next_obs[:, : self.obs_dim]

        # add_vae_belief=True
        add_vae_belief = False

        if use_tencoder:
            gru_h = augmented_obs[:, self.obs_dim + self.z_dim * 4 :]
            next_gru_h = augmented_next_obs[:, self.obs_dim + self.z_dim * 4 :]
            # with torch.no_grad():
            belief_mu, belief_logvar = self.task_encoder(gru_h)
        
            belief_next_mu, belief_next_logvar = self.task_encoder(next_gru_h)
            if add_vae_belief:
                vae_belief = augmented_obs[
                    :, self.obs_dim : self.obs_dim + self.z_dim * 2
                ]  # get current belief (t*b,_)
                vae_belief_next = augmented_next_obs[
                    :, self.obs_dim : self.obs_dim + self.z_dim * 2
                ]
                vae_belief_mu = vae_belief[:, : vae_belief.size(-1) // 2]
                vae_belief_logvar = vae_belief[:, vae_belief.size(-1) // 2 :]
                vae_belief_next_mu = vae_belief_next[:, : vae_belief.size(-1) // 2]
                vae_belief_next_logvar = vae_belief_next[:, vae_belief.size(-1) // 2 :]

                # belief_mu,belief_logvar=self.combine_gaussian_distributions(belief_mu,belief_logvar,vae_belief_mu,vae_belief_logvar,gru_h,None)
                # belief_next_mu,belief_next_logvar=self.combine_gaussian_distributions(belief_next_mu,belief_next_logvar,vae_belief_next_mu,vae_belief_next_logvar,next_gru_h,None)
                # with torch.no_grad():
                belief_mu, belief_logvar = self.vae.belief_encoder(
                    vae_belief_mu, vae_belief_logvar, belief_mu, belief_logvar
                )
                belief_next_mu, belief_next_logvar = self.vae.belief_encoder(
                    vae_belief_next_mu,
                    vae_belief_next_logvar,
                    belief_next_mu,
                    belief_next_logvar,
                )
        else:
            belief = augmented_obs[
                :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 4
            ]  # get current belief (t*b,_)
            belief_next = augmented_next_obs[
                :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 4
            ]

            belief_mu = belief[:, : belief.size(1) // 2]
            belief_logvar = belief[:, belief.size(1) // 2 :]
            belief_next_mu = belief_next[:, : belief_next.size(1) // 2]
            belief_next_logvar = belief_next[:, belief_next.size(1) // 2 :]

        z1 = self._sample_gaussian(
            belief_mu, belief_logvar
        )  # sample z from belief ,test: directly get current z
        z1_next = self._sample_gaussian(belief_next_mu, belief_next_logvar)

        add_kl = False
        if add_kl:
            # kl_divergences = (- 0.5 * (1 + belief_logvar - belief_mu.pow(2) - belief_logvar.exp()).sum(dim=-1))  + (- 0.5 * (1 + belief_next_logvar - belief_next_mu.pow(2) - belief_next_logvar.exp()).sum(dim=-1))
            kl_loss = self.compute_kl_loss(augmented_obs, augmented_next_obs)

        encode_state = self.state_encoder(state)
        # with torch.no_grad():
        encode_nextstate = self.state_encoder(next_state)
        augmented_state1 = torch.cat([encode_state, z1], dim=1)
        augmented_next_state1 = torch.cat([encode_nextstate, z1_next], dim=1)
        # sparse point robot  ignore next step state
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(
            augmented_state1, action
        )
        # encode_nextstate_detach=encode_nextstate.detach()
        diff = (
            pred_next_latent_mu - encode_nextstate
        ) / pred_next_latent_sigma  # detach?
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))

        pred_reward = self.reward_model(augmented_next_state1, None, action)
        # pred_reward = self.reward_model(augmented_state1,None,action)
        # reward_loss=F.mse_loss(pred_reward,reward, reduction='none')
        reward_loss = F.mse_loss(pred_reward, reward)
        if self.use_indyn:
            pred_action = self.invdyn_model(
                torch.cat([augmented_state1, augmented_next_state1], dim=-1)
            )
            action_loss = (action - pred_action).abs().mean()
            # action_loss=F.mse_loss(action - pred_action)
            total_loss = (
                loss + reward_loss + action_loss
            )  # + 2.0*kl_loss #+ kl_divergences.mean()
        else:
            total_loss = loss + reward_loss
        # total_loss=reward_loss  # only reward loss
        return total_loss

    def compute_rew_reconstruction_loss(self, action, next_state, rewards, rnn_output):
        with torch.no_grad():
            mean_shift, logvar_shift = self.task_encoder(rnn_output)
            encode_nextstate = self.state_encoder(next_state)
        z = self._sample_gaussian(mean_shift, logvar_shift)
        augmented_next_state = torch.cat([encode_nextstate, z], dim=-1)
        with torch.no_grad():
            pred_reward = self.reward_model(augmented_next_state, None, action)
        loss_rew = (pred_reward - rewards).pow(2).mean(dim=1)
        return loss_rew, pred_reward

    def update_latent_to_reality(self, augmented_obs, augmented_next_obs, action):
        # metric the latent belief(b1) and real belief(b2) by reward model(one step level)(trajectrory level)
        # state=augmented_obs[:,:self.obs_dim]
        next_state = augmented_next_obs[:, : self.obs_dim]
        encode_nextstate = self.state_encoder(next_state)

        belief = augmented_next_obs[
            :, self.obs_dim : self.obs_dim + self.z_dim * 2
        ]  # real belief(b1)
        belief_mu_ = belief[:, : self.z_dim]
        belief_logvar_ = belief[:, self.z_dim :]
        z_b2 = self._sample_gaussian(belief_mu_, belief_logvar_)
        gru_h = augmented_next_obs[:, self.obs_dim + self.z_dim * 4 :]
        mu_shift, logvar_shift = self.task_encoder(
            gru_h
        )  # latent belief(b2) using current task encoder
        belief_mu_, belief_logvar_ = self.vae.belief_encoder(
            belief_mu_, belief_logvar_, mu_shift, logvar_shift
        )
        z_b3 = self._sample_gaussian(belief_mu_, belief_logvar_)
        z_latent = self._sample_gaussian(mu_shift, logvar_shift)

        augmented_next_state_latent = torch.cat([encode_nextstate, z_latent], dim=-1)

        z_dist = F.smooth_l1_loss(z_b2, z_latent, reduction="none")

        with torch.no_grad():
            rew_pred_real = self.vae.reward_decoder_rec(z_b3, next_state, None, None)
        rew_pred_latent = self.reward_model(augmented_next_state_latent, None, action)

        rew_pred_dist = F.smooth_l1_loss(
            rew_pred_real, rew_pred_latent, reduction="none"
        )

        latet_reality_loss = F.mse_loss(torch.norm(z_dist), torch.norm(rew_pred_dist))

        return latet_reality_loss, z_dist.mean()

    def concat_encodestate_and_belief(
        self, augmented_obs
    ):  # get encodestate the concat with belief, to policy and critic
        belief = augmented_obs[:, self.obs_dim :]
        state = augmented_obs[:, : self.obs_dim]
        with torch.no_grad():
            encode_state = self.state_encoder(state)

        augmented_encode_obs = torch.cat((encode_state, belief), dim=-1)

        return augmented_encode_obs

    def concat_encodestate_and_belief_update_encoder(
        self, augmented_obs
    ):  # get encodestate the concat with belief, to policy and critic
        belief = augmented_obs[:, self.obs_dim :]
        state = augmented_obs[:, : self.obs_dim]
        encode_state = self.state_encoder(state)
        augmented_encode_obs = torch.cat((encode_state, belief), dim=-1)

        return augmented_encode_obs

    # def getobs_use_cur_taskencoder(self, augmented_obs, iter_, whether_next):
    #     state=augmented_obs[:,:self.obs_dim]
    #     belief=augmented_obs[:,self.obs_dim:self.obs_dim+self.z_dim*2]

    #     belief_mu_=belief[:,:self.z_dim]
    #     belief_logvar_=belief[:,self.z_dim:]

    #     gru_h=augmented_obs[:, self.obs_dim+self.z_dim*4:]
    #     # mu_shift,logvar_shift = self.task_encoder(gru_h)
    #     # with torch.no_grad():
    #     mu_shift,logvar_shift = self.task_encoder(gru_h)
    #     belief_mu,belief_logvar=self.vae.belief_encoder(belief_mu_,belief_logvar_,mu_shift,logvar_shift)
    #     # belief_mu,belief_logvar=self.vae.belief_encoder(belief_mu_,belief_logvar_,mu_shift,logvar_shift)
    #     # belief_mu,belief_logvar=self.bisim_belief_encoder(belief_mu_,belief_logvar_,mu_shift,logvar_shift)

    #     belief_mu,belief_logvar=self.adaptive_encoder(belief_mu,belief_logvar)

    #     augmented_encode_obs=torch.cat((state,belief_mu,belief_logvar),dim=-1)

    #     return augmented_encode_obs, belief_mu_, belief_logvar_, mu_shift,logvar_shift,gru_h

    def getobs_use_cur_taskencoder(self, augmented_obs, iter_, whether_next):
        state = augmented_obs[:, : self.obs_dim]
        belief = augmented_obs[:, self.obs_dim : self.obs_dim + self.z_dim * 2]

        belief_mu_ = belief[:, : self.z_dim]
        belief_logvar_ = belief[:, self.z_dim :]

        gru_h = augmented_obs[:, self.obs_dim + self.z_dim * 4 :]
        # mu_shift,logvar_shift = self.task_encoder(gru_h)
        # with torch.no_grad():
        mu_shift, logvar_shift = self.task_encoder(gru_h)

        if whether_next:
            mu_l_r, logvar_l_r = self.next_distshift_encoder(gru_h)
        else:
            mu_l_r, var_l_r = self.distshift_encoder(gru_h)  # latent to real

        # belief_mu = belief_mu_ + mu_shift + mu_l_r
        belief_mu, belief_var,mu2,var2 = self.gaussian_mixture_with_shift(belief_mu_,belief_logvar_, mu_shift, logvar_shift, mu_l_r, var_l_r)
        # belief_var = torch.exp(belief_logvar_)
        # var_shift = torch.exp(logvar_shift)
        # var_l_r = torch.exp(logvar_l_r) # 11.14
        # belief_var = belief_var + var_shift + var_l_r
        
        # var_l_r.data = torch.clamp(var_l_r.data, min=1e-6)
        
        if self.combine_outside:
            # belief_logvar = torch.log(belief_var+var_l_r )
            # belief_logvar = torch.log(belief_var) + var_l_r  #11.14
            # belief_logvar = torch.log(belief_var) + logvar_l_r # 11.15
            # belief_logvar = torch.log(belief_var) # 11.21
            # belief_logvar = torch.log(belief_var+ var_l_r)
            
            
            # belief_logvar = torch.log(belief_var+var_l_r)  # 11.22
            
            # ablation without offset
            belief_logvar = torch.log(belief_var)
            belief_logvar  = torch.clamp(belief_logvar , min=-20, max=2 )
            belief_mu=belief_mu
            # belief_mu = belief_mu + mu_l_r
        else:
            belief_logvar = torch.log(belief_var)
            belief_logvar  = torch.clamp(belief_logvar , min=-20, max=2)
        augmented_encode_obs = torch.cat((state, belief_mu, belief_logvar), dim=-1)
        # return (
        #     augmented_encode_obs,
        #     belief_mu_,    # b1_mu
        #     belief_logvar_, # b1_logvar
        #     mu_shift,
        #     logvar_shift,
        #     gru_h,
        # )
        return (
            augmented_encode_obs,
            belief_mu_,    # b1_mu
            belief_logvar_, # b1_logvar
            mu_shift,
            logvar_shift,
            mu_l_r, var_l_r,
            gru_h,
        )
        
        # return (
        #     augmented_encode_obs,    # b2-> b1+offset
        #     mu_shift,    # b1_mu
        #     logvar_shift, # b1_logvar
        #     mu1,
        #     var1,
        #     gru_h,
        # )
        # return (
        #     augmented_encode_obs,
        #     belief_mu_,    # b1_mu
        #     belief_logvar_, # b1_logvar
        #     mu2,
        #     var2,
        #     gru_h,
        # )
        # return (
        #     augmented_encode_obs,
        #     belief_mu_,    # b1_mu
        #     belief_logvar_, # b1_logvar
        #     belief_mu, belief_logvar,
        #     gru_h,
        # )

    # def getobs_use_cur_taskencoder(self, augmented_obs, iter_, whether_next):
    #     state=augmented_obs[:,:self.obs_dim]
    #     belief=augmented_obs[:,self.obs_dim:self.obs_dim+self.z_dim*2]

    #     belief_mu_=belief[:,:self.z_dim]
    #     belief_logvar_=belief[:,self.z_dim:]

    #     gru_h=augmented_obs[:, self.obs_dim+self.z_dim*4:]
    #     # mu_shift,logvar_shift = self.task_encoder(gru_h)
    #     # with torch.no_grad():
    #     # mu_shift,logvar_shift = self.task_encoder(gru_h)
    #     mu_shift=augmented_obs[:,self.obs_dim+self.z_dim*2:self.obs_dim+self.z_dim*3]
    #     logvar_shift=augmented_obs[:,self.obs_dim+self.z_dim*3:self.obs_dim+self.z_dim*4]
    #     # with torch.no_grad():
    #     #     belief_mu,belief_logvar=self.vae.belief_encoder(belief_mu_,belief_logvar_,mu_shift,logvar_shift)

    #     # # belief_mu,belief_logvar=self.bisim_belief_encoder(belief_mu_,belief_logvar_,mu_shift,logvar_shift)

    #     # belief_mu,belief_logvar=self.adaptive_encoder(belief_mu,belief_logvar)

    #     augmented_encode_obs=torch.cat((state,belief_mu_,belief_logvar_),dim=-1)

    #     return augmented_encode_obs, belief_mu_, belief_logvar_, mu_shift,logvar_shift,gru_h

    def getobs_notuse_cur_taskencoder(self, augmented_obs):

        state = augmented_obs[:, : self.obs_dim]
        belief = augmented_obs[:, self.obs_dim : self.obs_dim + self.z_dim * 2]

        belief_mu = belief[:, : self.z_dim]
        belief_logvar = belief[:, self.z_dim :]
        mu_shift = augmented_obs[
            :, self.obs_dim + self.z_dim * 2 : self.obs_dim + self.z_dim * 3
        ]
        logvar_shift = augmented_obs[
            :, self.obs_dim + self.z_dim * 3 : self.obs_dim + self.z_dim * 4
        ]

        gru_h = augmented_obs[:, self.obs_dim + self.z_dim * 4 :]

        # belief_mu,belief_logvar=self.overlap_dist(belief_mu,mu_shift,belief_logvar,logvar_shift)
        # belief_mu,belief_logvar=self.combine_gaussian_distributions_eval(belief_mu,belief_logvar,mu_shift,logvar_shift,gru_h)

        with torch.no_grad():
            # mu_l_r, logvar_l_r = self.distshift_encoder(gru_h)  # latent to real
            mu_l_r, var_l_r = self.distshift_encoder(gru_h)
            # mu_l_r, logvar_l_r = self.next_distshift_encoder(gru_h) 
        belief_mu, belief_var,mu2,var2=self.gaussian_mixture_with_shift(belief_mu, belief_logvar, mu_shift, logvar_shift, mu_l_r, var_l_r)
        # belief_mu = belief_mu + mu_shift + mu_l_r
        # belief_var = torch.exp(belief_logvar)
        # var_shift = torch.exp(logvar_shift)
        # var_l_r = torch.exp(logvar_l_r)   # 11.14
        # belief_var = belief_var + var_shift + var_l_r
        
        # var_l_r.data = torch.clamp(var_l_r.data, min=1e-6)

        if self.combine_outside:
            # belief_logvar = torch.log(belief_var+ var_l_r)
            # belief_logvar = torch.log(belief_var)+var_l_r # 11.14
            # belief_logvar = torch.log(belief_var)+logvar_l_r # 11.15
            # belief_logvar = torch.log(belief_var) # 11.21
            # belief_logvar = torch.log(belief_var+var_l_r)
            
            # belief_logvar = torch.log(belief_var+var_l_r)  # 11.22
            
            #ablation  without offset
            
            belief_logvar = torch.log(belief_var)
            belief_logvar  = torch.clamp(belief_logvar , min=-20, max=2)
            belief_mu=belief_mu
            # belief_mu = belief_mu + mu_l_r
        else:
            belief_logvar = torch.log(belief_var)
            belief_logvar  = torch.clamp(belief_logvar , min=-20, max=2)
        # belief_logvar=belief_logvar+logvar_shift
        # augmented_encode_obs=torch.cat((state,belief_mu,belief_logvar,mu_shift,logvar_shift),dim=-1)
        augmented_encode_obs = torch.cat((state, belief_mu, belief_logvar), dim=-1)
        return augmented_encode_obs

    # def getobs_notuse_cur_taskencoder(self,augmented_obs):

    #     state=augmented_obs[:,:self.obs_dim]
    #     belief=augmented_obs[:,self.obs_dim:self.obs_dim+self.z_dim*2]

    #     belief_mu=belief[:,:self.z_dim]
    #     belief_logvar=belief[:,self.z_dim:]
    #     mu_shift=augmented_obs[:,self.obs_dim+self.z_dim*2:self.obs_dim+self.z_dim*3]
    #     logvar_shift=augmented_obs[:,self.obs_dim+self.z_dim*3:self.obs_dim+self.z_dim*4]

    #     gru_h=augmented_obs[:,self.obs_dim+self.z_dim*4:]

    #     # belief_mu,belief_logvar=self.overlap_dist(belief_mu,mu_shift,belief_logvar,logvar_shift)
    #     # belief_mu,belief_logvar=self.combine_gaussian_distributions_eval(belief_mu,belief_logvar,mu_shift,logvar_shift,gru_h)

    #     # with torch.no_grad():
    #     #     belief_mu,belief_logvar=self.vae.belief_encoder(belief_mu,belief_logvar,mu_shift,logvar_shift)
    #     #     # belief_mu,belief_logvar=self.bisim_belief_encoder(belief_mu,belief_logvar,mu_shift,logvar_shift)
    #     #     belief_mu,belief_logvar=self.adaptive_encoder(belief_mu,belief_logvar)

    #     # belief_logvar=belief_logvar+logvar_shift
    #     augmented_encode_obs=torch.cat((state,belief_mu,belief_logvar),dim=-1)
    #     return augmented_encode_obs

    # augmented_obs[obs, task_mean, task_logvar, mu_shift, logvar_shift]
    def update(
        self,
        augmented_obs,
        action,
        reward,
        augmented_next_obs,
        done,
        origin_obs_shape,
        t_dim,
        b_dim,
        iter_,
        vae_recon_loss,
        **kwargs
    ):

        # v4
        # obs, belief_mu, belief_logvar, mu_shift, logvar_shift, rnn_output = (
        #     self.getobs_use_cur_taskencoder(augmented_obs, iter_, False)  # b2 -> b1
        # )
        
        obs, belief_mu, belief_logvar, mu_shift, logvar_shift,  mu_l_r, var_l_r,rnn_output = (
            self.getobs_use_cur_taskencoder(augmented_obs, iter_, False)  # b2 -> b1
        )
        
        # obs, mu_shift, logvar_shift, mu1, var1, rnn_output = (
        #     self.getobs_use_cur_taskencoder(augmented_obs, iter_, False)   #  b2 -> b1+offset
        # )
        # obs, belief_mu, belief_logvar, mu2, var2, rnn_output = (
        #     self.getobs_use_cur_taskencoder(augmented_obs, iter_, False)  # b2+offset -> b1
        # )
        # (
        #     next_obs,
        #     next_belief_mu,
        #     next_belief_logvar,
        #     next_mu_shift,
        #     next_logvar_shift,
        #     next_rnn_output,
        # ) = self.getobs_use_cur_taskencoder(augmented_next_obs, iter_, False)
        (
            next_obs,
            next_belief_mu,
            next_belief_logvar,
            next_mu_shift,
            next_logvar_shift,
             next_mu_l_r, next_var_l_r,
            next_rnn_output,
        ) = self.getobs_use_cur_taskencoder(augmented_next_obs, iter_, False)

        # obs=self.getobs_notuse_cur_taskencoder(augmented_obs)
        # next_obs=self.getobs_notuse_cur_taskencoder(augmented_next_obs)



        # computation of critic loss
        with torch.no_grad():
            next_action, _, _, next_log_prob = self.act(next_obs, return_log_prob=True)
            next_q1, next_q2 = self.critic_target(next_obs, next_action)
            min_next_q_target = (
                torch.min(next_q1, next_q2) - self.alpha_entropy * next_log_prob
            )
            q_target = reward + (1.0 - done) * self.gamma * min_next_q_target

        q1_pred, q2_pred = self.critic(obs, action)

        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        # use CQL loss for offline RL (Kumar et al, 2020)
        if self.use_cql:
            qf1_loss += torch.mean(
                self.alpha_cql
                * self.estimate_log_sum_exp_q(
                    self.critic.Q1, obs, N=10, action_space=kwargs["action_space"]
                )
                - q1_pred
            )
            qf2_loss += torch.mean(
                self.alpha_cql
                * self.estimate_log_sum_exp_q(
                    self.critic.Q2, obs, N=10, action_space=kwargs["action_space"]
                )
                - q2_pred
            )
        q_loss = qf1_loss + qf2_loss

        # --------------------update q networks
        self.critic_optimizer.zero_grad()
        # self.vae.belief_encoder_optimizer.zero_grad()
        # self.task_encoder_optimizer.zero_grad()  ###############
        # self.encoder_optimizer.zero_grad()  ###########  state encoder update by qloss
        # self.rho_optimizer.zero_grad()
        # self.bisim_belief_encoder_optimizer.zero_grad()
        # self.adaptive_encoder_optimizer.zero_grad()
        self.distshift_encoder_optimizer.zero_grad()
        q_loss.backward(retain_graph=True)
        if self.clip_grad_value is not None:
            self._clip_grads(self.critic)
        # torch.nn.utils.clip_grad_norm_(self.task_encoder.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.vae.belief_encoder.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.distshift_encoder_optimizer.step()
        # self.vae.belief_encoder_optimizer.step()
        # self.encoder_optimizer.step()#########
        # self.task_encoder_optimizer.step() ##############
        # self.rho_optimizer.step()
        # self.bisim_belief_encoder_optimizer.step()
        # self.adaptive_encoder_optimizer.step()
        # soft update
        self.soft_target_update()
        # self.soft_target_update_dist_shift()

        # ----------------------update encoder, transition, reward model
        encoder_decoder_recloss = self.update_transition_reward_decoder_cur(
            augmented_obs, action, augmented_next_obs, reward
        )
        beliefbisim_loss, z1, z1_next = self.update_context_encoder_cur(
            augmented_obs, action, augmented_next_obs, reward, t_dim, b_dim
        )
        
        #------------------------------------------------
        logvar_l_r=torch.log(var_l_r)
        logvar_l_r  = torch.clamp(logvar_l_r , min=-20, max=2 )
        kl_loss_2=self.kl_b1_b2(belief_mu, belief_logvar, mu_l_r, logvar_l_r)
        self.distshift_encoder_optimizer.zero_grad()
        kl_loss_2.backward(retain_graph=True)
        self.distshift_encoder_optimizer.step()
        #------------------------------------------------
        
        kl_loss = self.kl_b1_b2(belief_mu, belief_logvar, mu_shift, logvar_shift) 
        
        if self.update_latent_belief:
            latent_reality_loss, z_dist = self.update_latent_to_reality(
                augmented_obs, augmented_next_obs, action
            )
            latent_reality_loss = latent_reality_loss.requires_grad_()
        self.encoder_optimizer.zero_grad()
        self.vae.decoder_optimizer.zero_grad()
        self.task_encoder_optimizer.zero_grad()
        if self.use_indyn:
            self.invdyn_optimizer.zero_grad()
        loss = (
            # encoder_decoder_recloss + beliefbisim_loss + iter_/100 * kl_loss + q_loss
            encoder_decoder_recloss + beliefbisim_loss + kl_loss  
        )  # + q_loss   # qloss
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.task_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_value_(self.vae.state_decoder.parameters(), 0.2)  ################
        self.encoder_optimizer.step()
        self.vae.decoder_optimizer.step()  # update   vae decoder optimuzer
        if self.use_indyn:
            self.invdyn_optimizer.step()
        self.task_encoder_optimizer.step()

        # computation of actor loss---------------------------------------
        # obs,belief_mu,belief_logvar,mu_shift,logvar_shift,rnn_output=self.getobs_use_cur_taskencoder(augmented_obs,iter_,False)
        obs,belief_mu,belief_logvar, mu_shift, logvar_shift,mu_l_r, var_l_r,rnn_output=self.getobs_use_cur_taskencoder(augmented_obs,iter_,False)
        new_action, _, _, log_prob = self.act(obs, return_log_prob=True)
        min_q_new_actions = self._min_q(
            obs, new_action
        )  # critic_obs or policy_obs?  #############################
        policy_loss = ((self.alpha_entropy * log_prob) - min_q_new_actions).mean()

        # ----------------------update policy network
        self.policy_optim.zero_grad()
        policy_loss = policy_loss.requires_grad_()
        policy_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.policy)
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -(
                self.log_alpha_entropy * (log_prob + self.target_entropy).detach()
            ).mean()
            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp()
            # alpha_entropy_tlogs = self.alpha_entropy.clone()    # For TensorboardX logs
        else:
            alpha_entropy_loss = torch.tensor(0.0).to(ptu.device)
            # alpha_entropy_tlogs = torch.tensor(self.alpha_entropy)  # For TensorboardX logs

        return {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_entropy_loss": alpha_entropy_loss.item(),
            "sac_encoder_decoder_loss": beliefbisim_loss.item(),
        }

    def perm_tensor(self, h, origin_obs_shape, perm):
        h = h.view(origin_obs_shape[0], origin_obs_shape[1], -1)
        # perm1_2 = torch.randperm(origin_obs_shape[0])
        h2 = h[perm]
        h2 = h2.view(origin_obs_shape[0] * origin_obs_shape[1], -1)
        return h2

    def _min_q(self, obs, action):
        q1, q2 = self.critic(obs, action)
        min_q = torch.min(q1, q2)
        return min_q

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
    
    def soft_target_update_dist_shift(self):
        ptu.soft_update_from_to(self.distshift_encoder, self.next_distshift_encoder, self.tau)
    
    
    def _clip_grads(self, net):
        for p in net.parameters():
            p.grad.data.clamp_(-self.clip_grad_value, self.clip_grad_value)

    def estimate_log_sum_exp_q(self, qf, obs, N, action_space):
        """
            estimate log(sum(exp(Q))) for CQL objective
        :param qf: Q function
        :param obs: state batch from buffer (s~D)
        :param N: number of actions to sample for estimation
        :param action_space: space of actions -- for uniform sampling
        :return:
        """
        batch_size = obs.shape[0]
        obs_rep = obs.repeat(N, 1)

        # draw actions at uniform
        random_actions = ptu.FloatTensor(
            np.vstack([action_space.sample() for _ in range(N)])
        )
        random_actions = torch.repeat_interleave(random_actions, batch_size, dim=0)
        unif_a = 1 / np.prod(
            action_space.high - action_space.low
        )  # uniform density over action space

        # draw actions from current policy
        with torch.no_grad():
            policy_actions, _, _, policy_log_probs = self.act(
                obs_rep, return_log_prob=True
            )

        exp_q_unif = qf(obs_rep, random_actions) / unif_a
        exp_q_policy = qf(obs_rep, policy_actions) / torch.exp(policy_log_probs)
        log_sum_exp = torch.log(
            0.5
            * torch.mean((exp_q_unif + exp_q_policy).reshape(N, batch_size, -1), dim=0)
        )

        return log_sum_exp

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).half()
            return eps.mul(std).add_(mu)
        else:
            if logvar.shape[0] > 1:
                mu = mu.unsqueeze(0)
                logvar = logvar.unsqueeze(0)
            if logvar.dim() > 2:  # if 3 dims, first must be 1
                assert logvar.shape[0] == 1, "error in dimensions!"
                std = torch.exp(0.5 * logvar).repeat(num, 1, 1).half()
                eps = torch.randn_like(std).half()
                mu = mu.repeat(num, 1, 1).half()
            else:
                std = torch.exp(0.5 * logvar).repeat(num, 1).half()
                eps = torch.randn_like(std).half()
                mu = mu.repeat(num, 1).half()
            torch.cuda.empty_cache()
            return eps.mul(std).add_(mu)

    def gaussian_mixture_with_shift(self, mu1, logvar1, mu2, logvar2, mu_offset, var_offset, w1=0.5, w2=0.5):
        # if not isinstance(w1, (float, int, torch.Tensor)):
        #     w1 = float(w1)
        # if not isinstance(w2, (float, int, torch.Tensor)):
        #     w2 = float(w2)
    
        # # 确保 w1 和 w2 是张量
        # w1 = torch.tensor(w1, dtype=torch.float32)
        # w2 = torch.tensor(w2, dtype=torch.float32)
   
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        # var_offset=torch.exp(logvar_offset)
        with_offset=True  
        
        if not self.combine_outside:
            if with_offset: 
                mu1=mu1+mu_offset
                var1=var1+var_offset
                mu2 = mu2 + mu_offset
                var2 = var2 + var_offset
            else:
                pass
        # var2 = torch.clamp(var2, min=1e-6)   
        mu_m = w1 * mu1 + w2 * mu2  
        
        sigma_m2 = w1 * (var1 + mu1**2) + w2 * (var2 + mu2**2) - mu_m**2
        
        # sigma_m2 = w1 * var1  + w2 * var2
        sigma_m2 = torch.clamp(sigma_m2, min=1e-6)
        # logvar2 = torch.log(var2)
        return mu_m, sigma_m2 ,mu2,var2

       



class Critic(nn.Module):
    def __init__(
        self,
        q1_network,
        q2_network,
        # encoder,
    ):
        super().__init__()
        # self.encoder = encoder

        self.Q1 = q1_network
        self.Q2 = q2_network
        self.outputs = dict()

    def forward(self, obs, action):
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2
