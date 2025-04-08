# SimBelief

Source Code for 'Learning Task Belief Similarity with Latent Dynamics for Meta-Reinforcement Learning'

## Installation

To set up the environment, you can use the provided `<span>environment.yml</span>` file for Conda or install dependencies from `<span>requirements.txt</span>`.

```
conda env create -f environment.yml
conda activate simbelief
```

```
pip install -r requirements.txt
```

## Training

To train an online agent, run:

```
python online_training.py --env-type env_name --seed seed_numer
```

## Citation

@inproceedings{
zhang2025learning,
title={Learning Task Belief Similarity with Latent Dynamics for Meta-Reinforcement Learning},
author={Menglong Zhang and Fuyuan Qian and Quanying Liu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=5YbuOTUFQ4}
}
