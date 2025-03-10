# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import torch.nn as nn


class DeterministicTransitionModel(nn.Module):

    def __init__(self, encoder_feature_dim, action_shape, layer_width, encoder_max_norm=None):
        super().__init__()
        self.fc = nn. Linear(encoder_feature_dim + action_shape, layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.max_norm = encoder_max_norm
        print("Deterministic transition model chosen.")

    def forward(self, x, normalize=True):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        if self.max_norm and normalize:
            mu = self.normalize(mu)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        return mu
    
    def normalize(self, x):
        norms = x.norm(dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max

        return x


class ProbabilisticTransitionModel(nn.Module):

    def __init__(self, encoder_feature_dim, action_shape, layer_width, announce=True, max_sigma=1e1, min_sigma=1e-4,
                 encoder_max_norm=None):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim + action_shape, layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)
        if announce:
            print("Probabilistic transition model chosen.")

        self.max_norm = encoder_max_norm

    def forward(self, x, normalize=True):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        if self.max_norm and normalize:
            mu = self.normalize(mu)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        ret = mu + sigma * eps
        if self.max_norm:
            ret = self.normalize(ret)
            # WARNING: not adjusting for non-linear change in distribution.
        return ret

    def normalize(self, x):
        norms = x.norm(dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max

        return x
    



class EnsembleOfProbabilisticTransitionModels(object):

    def __init__(self, encoder_feature_dim, action_shape, layer_width, ensemble_size=5):
        self.models = [ProbabilisticTransitionModel(encoder_feature_dim, action_shape, layer_width, announce=False)
                       for _ in range(ensemble_size)]
        print("Ensemble of probabilistic transition models chosen.")

    def __call__(self, x):
        mu_sigma_list = [model.forward(x) for model in self.models]
        mus, sigmas = zip(*mu_sigma_list)
        mus, sigmas = torch.stack(mus), torch.stack(sigmas)
        return mus, sigmas

    def sample_prediction(self, x):
        model = random.choice(self.models)
        return model.sample_prediction(x)

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def parameters(self):
        list_of_parameters = [list(model.parameters()) for model in self.models]
        parameters = [p for ps in list_of_parameters for p in ps]
        return parameters


_AVAILABLE_TRANSITION_MODELS = {'': DeterministicTransitionModel,
                                'deterministic': DeterministicTransitionModel,
                                'probabilistic': ProbabilisticTransitionModel,
                                'ensemble': EnsembleOfProbabilisticTransitionModels}


def make_transition_model(transition_model_type, encoder_feature_dim, action_shape, layer_width=512, encoder_max_norm=None):
    assert transition_model_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[transition_model_type](
        encoder_feature_dim, action_shape, layer_width, encoder_max_norm=encoder_max_norm
    )
