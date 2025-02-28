import json
import datetime
import time
import os
from tensorboardX import SummaryWriter
from torchkit import pytorch_utils as ptu

import numpy as np
import torch
from utils import evaluation as utl_eval


class TBLogger:
    def __init__(self, args):

        # initialise name of the file (optional(prefix) + seed + start time)
        cql_ext = '_cql' if 'use_cql' in args and args.use_cql else ''
        if hasattr(args, 'output_file_prefix'):
            self.output_name = args.output_file_prefix + cql_ext + \
                               '__' + str(args.seed) + '__' + \
                               datetime.datetime.now().strftime('%d_%m_%H_%M_%S')
        else:
            self.output_name = str(args.seed) + '__' + datetime.datetime.now().strftime('%d_%m_%H_%M_%S')

        # get path to log directory (and create it if necessary)
        try:
            log_dir = args.results_log_dir
        except AttributeError:
            log_dir = args['results_log_dir']

        if log_dir is None:
            log_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
            log_dir = os.path.join(log_dir, 'logs')

        if not os.path.exists(log_dir):
            try:
                os.mkdir(log_dir)
            except:
                dir_path_head, dir_path_tail = os.path.split(log_dir)
                if len(dir_path_tail) == 0:
                    dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                os.mkdir(dir_path_head)
                os.mkdir(log_dir)

        # create a subdirectory for the environment
        try:
            env_dir = os.path.join(log_dir, '{}'.format(args.env_name))
        except:
            env_dir = os.path.join(log_dir, '{}'.format(args["env_name"]))
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)

        # create a subdirectory for the exp_label (usually the method name)
        # exp_dir = os.path.join(env_dir, exp_label)
        # if not os.path.exists(exp_dir):
        #     os.makedirs(exp_dir)

        # finally, get full path of where results are stored
        self.full_output_folder = os.path.join(env_dir, self.output_name)

        self.writer = SummaryWriter(self.full_output_folder)

        print('logging under', self.full_output_folder)

        with open(os.path.join(self.full_output_folder, 'online_config.json'), 'w') as f:
            try:
                config = {k: v for (k, v) in vars(args).items() if k != 'device'}
            except:
                config = args
            config.update(device=ptu.device.type)
            json.dump(config, f, indent=2)


def log(self, iteration, train_stats):
        # --- save models ---
        if iteration % self.args.save_interval == 0:
            save_path = os.path.join(self.tb_logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(self.agent.state_dict(), os.path.join(save_path, "agent{0}.pt".format(iteration)))
            torch.save(self.vae.encoder.state_dict(), os.path.join(save_path, "encoder{0}.pt".format(iteration)))
            if self.vae.reward_decoder is not None:
                torch.save(self.vae.reward_decoder.state_dict(), os.path.join(save_path, "reward_decoder{0}.pt".format(iteration)))
            if self.vae.state_decoder is not None:
                torch.save(self.vae.state_decoder.state_dict(), os.path.join(save_path, "state_decoder{0}.pt".format(iteration)))
            if self.vae.task_decoder is not None:
                torch.save(self.vae.task_decoder.state_dict(), os.path.join(save_path, "task_decoder{0}.pt".format(iteration)))

        # evaluate to get more stats
        if self.args.policy == 'dqn':
            # get stats on train tasks
            returns_train, success_rate_train, values, reward_preds, \
            task_samples, task_means, task_logvars = self.evaluate(self.train_tasks[:len(self.eval_tasks)])
            returns_eval, success_rate_eval, values, reward_preds, \
            task_samples, task_means, task_logvars = self.evaluate(self.eval_tasks)
        else:
            # get stats on train tasks
            returns_train, success_rate_train, log_probs, observations, rewards_train, reward_preds_train, \
            task_samples, task_means, task_logvars = self.evaluate(self.train_tasks[:len(self.eval_tasks)])
            returns_eval, success_rate_eval, _, observations_eval, rewards_eval, reward_preds_eval, \
            _, _, _ = self.evaluate(self.eval_tasks)

        if self.args.log_tensorboard:
            # --- log training  ---
            if self.args.policy == 'dqn':
                # for i, task in enumerate(self.eval_tasks):
                for i, task in enumerate(self.train_tasks[:5]):
                    self.tb_logger.writer.add_figure('rewards_pred_task_{}/prior'.format(i),
                                                     utl_eval.vis_rew_pred(self.args, reward_preds[i, 0].round(2),
                                                                           self.env.goals[task]),
                                                     self._n_env_steps_total)
                    self.tb_logger.writer.add_figure('rewards_pred_task_{}/halfway'.format(i),
                                                     utl_eval.vis_rew_pred(self.args, reward_preds[i, int(np.ceil(reward_preds.shape[1] / 2))].round(2),
                                                                           self.env.goals[task]),
                                                     self._n_env_steps_total)
                    self.tb_logger.writer.add_figure('rewards_pred_task_{}/final'.format(i),
                                                     utl_eval.vis_rew_pred(self.args, reward_preds[i, -1].round(2),
                                                                           self.env.goals[task]),
                                                     self._n_env_steps_total)
            else:
                for i, task in enumerate(self.train_tasks[:5]):
                    self.env.reset(task)
                    # self.tb_logger.writer.add_figure('policy_vis_train/task_{}'.format(i),
                    #                                  utl_eval.plot_rollouts(observations[i, :], self.env),
                    #                                  self._n_env_steps_total)
                    # # sample batch
                    # obs, _, _, _, _ = self.sample_rl_batch(tasks=[task],
                    #                                        batch_size=self.policy_storage.task_buffers[task].size())
                    # self.tb_logger.writer.add_figure('state_space_coverage/task_{}'.format(i),
                    #                                  utl_eval.plot_visited_states(ptu.get_numpy(obs[0]), self.env),
                    #                                  self._n_env_steps_total)
                    self.tb_logger.writer.add_figure('reward_prediction_train/task_{}'.format(i),
                                                     utl_eval.plot_rew_pred_vs_rew(rewards_train[i, :],
                                                                                   reward_preds_train[i, :]),
                                                     self._n_env_steps_total)
                    
                    # self.tb_logger.writer.add_figure('Plot mean/variance/pred_rewards over time/task_{}'.format(i),
                    #                                  utl_eval.plot_latents(task_means[i, :],task_logvars[i, :],reward_preds_train[i, :],2,self.args.max_trajectory_len),
                    #                                  self._n_env_steps_total)
                    
                    # self.tb_logger.writer.add_figure('belief_halfcircle/task_{}'.format(i),
                    #                                  utl_eval.plot_discretized_belief_halfcircle(reward_preds_train[i, :], center_points, self.env, observations),
                    #                                  self._n_env_steps_total)
                    
                    # self.tb_logger.writer.add_figure('visited_states/task_{}'.format(i),
                    #                                  utl_eval.plot_visited_states(observations[i], self.env),
                    #                                  self._n_env_steps_total)



                for i, task in enumerate(self.eval_tasks[:5]):
                    self.env.reset(task)
                    # self.tb_logger.writer.add_figure('policy_vis_eval/task_{}'.format(i),
                    #                                  utl_eval.plot_rollouts(observations_eval[i, :], self.env),
                    #                                  self._n_env_steps_total)
                    self.tb_logger.writer.add_figure('reward_prediction_eval/task_{}'.format(i),
                                                     utl_eval.plot_rew_pred_vs_rew(rewards_eval[i, :],
                                                                                   reward_preds_eval[i, :]),
                                                     self._n_env_steps_total)
            # some metrics
            self.tb_logger.writer.add_scalar('metrics/successes_in_buffer',
                                             self._successes_in_buffer / self._n_env_steps_total,
                                             self._n_env_steps_total)

            if self.args.max_rollouts_per_task > 1:
                for episode_idx in range(self.args.max_rollouts_per_task):
                    self.tb_logger.writer.add_scalar('returns_multi_episode/episode_{}'.
                                                     format(episode_idx + 1),
                                                     np.mean(returns_train[:, episode_idx]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns_multi_episode/sum',
                                                 np.mean(np.sum(returns_train, axis=-1)),
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns_multi_episode/success_rate',
                                                 np.mean(success_rate_train),
                                                 self._n_env_steps_total)
                if self.args.policy != 'dqn':
                    self.tb_logger.writer.add_scalar('returns_multi_episode/sum_eval',
                                                     np.mean(np.sum(returns_eval, axis=-1)),
                                                     self._n_env_steps_total)
                    self.tb_logger.writer.add_scalar('returns_multi_episode/success_rate_eval',
                                                     np.mean(success_rate_eval),
                                                     self._n_env_steps_total)
            else:
                # self.tb_logger.writer.add_scalar('returns/returns_mean', np.mean(returns),
                #                                  self._n_env_steps_total)
                # self.tb_logger.writer.add_scalar('returns/returns_std', np.std(returns),
                #                                  self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns/returns_mean_train', np.mean(returns_train),
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns/returns_std_train', np.std(returns_train),
                                                 self._n_env_steps_total)
                # self.tb_logger.writer.add_scalar('returns/success_rate', np.mean(success_rate),
                #                                  self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns/success_rate_train', np.mean(success_rate_train),
                                                 self._n_env_steps_total)
            # encoder
            self.tb_logger.writer.add_scalar('encoder/task_embedding_init', task_samples[:, 0].mean(), self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('encoder/task_mu_init', task_means[:, 0].mean(), self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('encoder/task_logvar_init', task_logvars[:, 0].mean(), self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('encoder/task_embedding_halfway', task_samples[:, int(task_samples.shape[-1]/2)].mean(), self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('encoder/task_mu_halfway', task_means[:, int(task_means.shape[-1]/2)].mean(), self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('encoder/task_logvar_halfway', task_logvars[:, int(task_logvars.shape[-1]/2)].mean(), self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('encoder/task_embedding_final', task_samples[:, -1].mean(), self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('encoder/task_mu_final', task_means[:, -1].mean(), self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('encoder/task_logvar_final', task_logvars[:, -1].mean(), self._n_env_steps_total)

            # policy
            if self.args.policy == 'dqn':
                self.tb_logger.writer.add_scalar('policy/value_init', np.mean(values[:, 0]), self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('policy/value_halfway', np.mean(values[:, int(values.shape[-1]/2)]), self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('policy/value_final', np.mean(values[:, -1]), self._n_env_steps_total)

                self.tb_logger.writer.add_scalar('policy/exploration_epsilon', self.agent.eps, self._n_env_steps_total)
                # RL losses
                self.tb_logger.writer.add_scalar('rl_losses/qf_loss_vs_n_updates', train_stats['qf_loss'],
                                                 self._n_rl_update_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/qf_loss_vs_n_env_steps', train_stats['qf_loss'],
                                                 self._n_env_steps_total)
            else:
                self.tb_logger.writer.add_scalar('policy/log_prob', np.mean(log_probs), self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/qf1_loss', train_stats['qf1_loss'],
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/qf2_loss', train_stats['qf2_loss'],
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/policy_loss', train_stats['policy_loss'],
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/alpha_entropy_loss', train_stats['alpha_entropy_loss'],
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/sac_encoder_decoder_loss', train_stats['sac_encoder_decoder_loss'],
                                                 self._n_env_steps_total)

            # VAE losses
            self.tb_logger.writer.add_scalar('vae_losses/vae_loss', train_stats['vae_loss'],
                                             self._n_env_steps_total)
            self.tb_logger.writer.add_scalar('vae_losses/kl_loss', train_stats['kl_loss'],
                                             self._n_env_steps_total)
            # self.tb_logger.writer.add_scalar('vae_losses/similarity_loss',
            #                                      train_stats['similarity_loss'],
            #                                      self._n_env_steps_total)
            if self.vae.reward_decoder is not None:
                self.tb_logger.writer.add_scalar('vae_losses/reward_rec_loss',
                                                 train_stats['rew_loss'],
                                                 self._n_env_steps_total)
            if self.vae.state_decoder is not None:
                self.tb_logger.writer.add_scalar('vae_losses/state_rec_loss',
                                                 train_stats['state_loss'],
                                                 self._n_env_steps_total)
            if self.vae.task_decoder is not None:
                self.tb_logger.writer.add_scalar('vae_losses/task_rec_loss',
                                                 train_stats['task_loss'],
                                                 self._n_env_steps_total)

            # weights and gradients
            if self.args.policy == 'dqn':
                self.tb_logger.writer.add_scalar('weights/q_network',
                                                 list(self.agent.qf.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q_network',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/q_target',
                                                 list(self.agent.target_qf.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.target_qf.parameters())[0].grad is not None:
                    param_list = list(self.agent.target_qf.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q_target',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
            else:
                self.tb_logger.writer.add_scalar('weights/q1_network',
                                                 list(self.agent.qf1.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf1.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf1.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q1_network',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/q1_target',
                                                 list(self.agent.qf1_target.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf1_target.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf1_target.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q1_target',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/q2_network',
                                                 list(self.agent.qf2.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf2.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf2.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q2_network',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/q2_target',
                                                 list(self.agent.qf2_target.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf2_target.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf2_target.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q2_target',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/policy',
                                                 list(self.agent.policy.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.policy.parameters())[0].grad is not None:
                    param_list = list(self.agent.policy.parameters())
                    self.tb_logger.writer.add_scalar('gradients/policy',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)

            self.tb_logger.writer.add_scalar('weights/encoder',
                                             list(self.vae.encoder.parameters())[0].mean(),
                                             self._n_env_steps_total)
            if list(self.vae.encoder.parameters())[0].grad is not None:
                param_list = list(self.vae.encoder.parameters())
                self.tb_logger.writer.add_scalar('gradients/encoder',
                                                 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                 self._n_env_steps_total)

            if self.vae.reward_decoder is not None:
                self.tb_logger.writer.add_scalar('weights/reward_decoder',
                                                 list(self.vae.reward_decoder.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.vae.reward_decoder.parameters())[0].grad is not None:
                    param_list = list(self.vae.reward_decoder.parameters())
                    self.tb_logger.writer.add_scalar('gradients/reward_decoder',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
            if self.vae.state_decoder is not None:
                self.tb_logger.writer.add_scalar('weights/state_decoder',
                                                 list(self.vae.state_decoder.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.vae.state_decoder.parameters())[0].grad is not None:
                    param_list = list(self.vae.state_decoder.parameters())
                    self.tb_logger.writer.add_scalar('gradients/state_decoder',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
            if self.vae.task_decoder is not None:
                self.tb_logger.writer.add_scalar('weights/task_decoder',
                                                 list(self.vae.task_decoder.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.vae.task_decoder.parameters())[0].grad is not None:
                    param_list = list(self.vae.task_decoder.parameters())
                    self.tb_logger.writer.add_scalar('gradients/task_decoder',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)

        # output to user
        # print("Iteration -- {:3d}, Num. RL updates -- {:6d}, Elapsed time {:5d}[s]".
        #       format(iteration,
        #              self._n_rl_update_steps_total,
        #              int(time.time() - self._start_time)))
        print("Iteration -- {}, Success rate train -- {:.3f}, Success rate eval.-- {:.3f}, "
              "Avg. return train -- {:.3f}, Avg. return eval. -- {:.3f}, Elapsed time {:5d}[s]"
              .format(iteration, np.mean(success_rate_train),
                      np.mean(success_rate_eval), np.mean(np.sum(returns_train, axis=-1)),
                      np.mean(np.sum(returns_eval, axis=-1)),
                      int(time.time() - self._start_time)))

