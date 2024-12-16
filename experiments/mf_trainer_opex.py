# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import absl.flags
import gym
import numpy as np
import tqdm
from flax import linen as nn
import jax
import jax.numpy as jnp
from functools import partial
import algos
from algos.mf_algos.model import (
  ClipGaussianPolicy,
  DirectMappingPolicy,
  FullyConnectedQFunction,
  FullyConnectedVFunction,
  ResClipGaussianPolicy,
  ResDirectMappingPolicy,
  ResQFunction,
  ResTanhGaussianPolicy,
  ResVFunction,
  SamplerPolicy,
  TanhGaussianPolicy,
)
from core.core_api import Trainer
from algos.mf_algos.data import (
  Dataset, DM2Gym, RandSampler, RLUPDataset, BC_RLUPDataset
)
from experiments.args import FLAGS_DEF
from experiments.constants import (
  ALGO,
  ALGO_MAP,
  DATASET,
  DATASET_ABBR_MAP,
  DATASET_MAP,
  ENV,
  ENV_MAP,
  ENV_REW_CLIP_VAL,
  ENVNAME_MAP,
)
from algos.mf_algos.data.replay_buffer import get_d4rl_dataset
from algos.mf_algos.data.sampler import TrajSampler
from utilities.jax_utils import batch_to_jax
from utilities.utils import (
  Timer,
  WandBLogger,
  get_user_flags,
  norm_obs,
  prefix_metrics,
  set_random_seed,
)
from viskit.logging import logger, setup_logger
from flax.training import checkpoints

class MFTrainer_OPEX(Trainer):

  def __init__(self):
    self._cfgs = absl.flags.FLAGS
    self._algo = getattr(algos, self._cfgs.algo_cfg.name)
    self._algo_type = ALGO_MAP[self._cfgs.algo_cfg.name]
    self._training_cfgs = self._cfgs.algo_cfg.training
    self._agent_cfgs = self._cfgs.algo_cfg.agent
    self.opex_beta = self._cfgs.opex_beta
    self.num_steps = self._cfgs.num_steps
    # self.decrease_beta = self._cfgs.decrease_beta
    self.norm_grad = self._cfgs.norm_grad
    self.test_mode = self._cfgs.test_mode
    self.use_opex = self._cfgs.use_opex
    self._setup_env()

    self._variant = get_user_flags(self._cfgs, FLAGS_DEF)
    algo_cfgs = dict(
      name=self._cfgs.algo_cfg.name,
      type=self._cfgs.algo_cfg.type,
      training=dict(self._training_cfgs),
      agent=dict(self._agent_cfgs),
    )
    self._variant.update(dict(algo_cfg=algo_cfgs,))

    self._obs_mean: float = None
    self._obs_std: float = None
    self._obs_clip: float = None

    self._eval_sampler: TrajSampler = None
    self._observation_dim: int = None
    self._action_dim: int = None

    self._wandb_logger: WandBLogger = None
    self._dataset: Dataset = None
    self._policy: nn.Module = None
    self._qf: nn.Module = None
    self._vf: nn.Module = None
    self._agent: object = None
    self._sampler_policy: SamplerPolicy = None

  # def update_opex_beta(self, epoch):
  #   """
  #   OPEX Beta 값을 업데이트하는 함수.
  #   """
  #   if self.decrease_beta:
  #       epoch = max(0, min(epoch, self._training_cfgs.n_epochs)) 
  #       # 선형 감소
  #       self.opex_beta = self.opex_beta * (1 - epoch / self._training_cfgs.n_epochs)
  #       # 지수 감소 (원하면 주석 해제)
  #       # self.opex_beta = self.opex_beta * np.exp(-epoch / self._cfgs.decay_rate)

  def flags_to_dict(flags):
    return {flag.name: flag.value for flag in flags.flags_by_module_dict().values()}

  def _setup_env(self):
    # get high level env
    env_name_full = self._cfgs.env
    for scenario_name in ENV_MAP:
      if scenario_name in env_name_full:
        self._env = ENV_MAP[scenario_name]
        break
    else:
      raise NotImplementedError
  
  def train(self):
    if self._algo_type in [ALGO.Onestep, ALGO.IQL_Onestep]:
      self._train_onestep()
    else:
      self._train()

  def _train(self):
    self._setup()

    viskit_metrics = {}
    avg_norm_returns = []
    metrics = {}

    if self.test_mode : 
      if self.use_opex : 
        self._eval_opex(epoch, metrics, avg_norm_returns)
        metrics["eval_opex_time"] = eval_opex_timer()
      else : 
        self._eval(epoch, metrics, avg_norm_returns)
        metrics["eval_time"] = eval_timer()

      print(metrics)
      return 

    for epoch in range(self._training_cfgs.n_epochs):
      metrics = {"epoch": epoch}
      train_results = {}

      with Timer() as train_timer:
        total_rewards = []
        total_lengths = []
        total_normalized_returns = []
        dones = []

        for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
          batch = batch_to_jax(self._dataset.sample())
          cur_step = epoch * self._cfgs.n_train_step_per_epoch + _
          update_target_policy = False
          if cur_step > self._training_cfgs.target_policy_warmup:
            if (
              cur_step + 1
            ) % self._training_cfgs.target_policy_update_interval == 0:
              update_target_policy = True
          step_metrics = self._agent.train(batch, update_target_policy)
          metrics.update(
            prefix_metrics(
              step_metrics, "agent"
            )
          )
          
          for key, value in step_metrics.items():
            if key not in train_results:
              train_results[key] = []
            train_results[key].append(value)

          # Collect statistics for train_results
          total_rewards.append(np.sum(batch["rewards"]))
          total_lengths.append(len(batch["rewards"]))
          total_normalized_returns.append(
              self._eval_sampler.env.get_normalized_score(np.sum(batch["rewards"]))
          )
          dones.append(np.sum(batch["dones"]))
      
        # Aggregate collected statistics
        train_results["train_average_return"] = np.mean(total_rewards)
        train_results["train_average_traj_length"] = np.mean(total_lengths)
        train_results["train_average_normalizd_return"] = np.mean(total_normalized_returns)
        avg_norm_returns.append(train_results["train_average_normalizd_return"])
        train_results['train_average_10_normalized_return'] = np.mean(
            avg_norm_returns[-10:]
        )
        train_results["train_best_normalized_return"] = max(avg_norm_returns)
        train_results["train_done"] = np.mean(dones)

        metrics.update(train_results)

        if epoch % 500 == 0:
          if self._cfgs.save_model:
                  save_data = {
                  "agent": self._agent,
                  #"cfgs": self._cfgs,
                  "epoch": epoch
                  }
                  self._wandb_logger.save_pickle(save_data, f"model_{epoch}.pkl")

      # with Timer() as eval_timer:
      #   if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:
      #     self._eval(epoch, metrics, avg_norm_returns)

      with Timer() as eval_opex_timer:
        if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:
          self._eval_opex(epoch, metrics, avg_norm_returns)

      metrics["train_time"] = train_timer()
      #metrics["eval_time"] = eval_timer()
      metrics["eval_opex_time"] = eval_opex_timer()
      #metrics["epoch_time"] = train_timer() + eval_timer() + eval_opex_timer()
      metrics["epoch_time"] = train_timer() + eval_opex_timer()
      self._log(viskit_metrics, metrics)

    # save model
    if self._cfgs.save_model:
      save_data = {"agent": self._agent, #"cfgs": self._cfgs, 
                   "epoch": epoch}
      self._wandb_logger.save_pickle(save_data, "model_final.pkl")

  def _train_onestep(self):
    self._setup()

    viskit_metrics = {}
    avg_norm_returns = []
    if self._algo_type == ALGO.Onestep:
      policy_name = 'pi'
    elif self._algo_type == ALGO.IQL_Onestep:
      policy_name = 'policy'

    for epoch in range(self._training_cfgs.n_epochs):
      metrics = {"epoch": epoch}
      if epoch == self._training_cfgs.beta_n_epochs + self._training_cfgs.qf_n_epochs:
        if self._training_cfgs.beta_n_epochs > 0:
          self._agent.init_pi() # init pi by beta
        self._eval(
          epoch, metrics, avg_norm_returns, policy_name=policy_name, log_prefix=True
        )
        self._eval_opex(
          epoch, metrics, avg_norm_returns, policy_name=policy_name, log_prefix=True
        )
      with Timer() as train_timer:
        for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
          train_beta, train_pi = False, False
          batch = batch_to_jax(self._dataset.sample())
          if epoch < self._training_cfgs.beta_n_epochs:
            # train behavior policy
            train_beta = True
            metrics.update(prefix_metrics(self._agent.train_beta(batch), "beta"))
          elif epoch < self._training_cfgs.beta_n_epochs + self._training_cfgs.qf_n_epochs:
            # train qf
            metrics.update(prefix_metrics(self._agent.train_qf(batch), "qf"))
          else:
            # train pi
            train_pi = True
            metrics.update(prefix_metrics(self._agent.train_pi(batch), policy_name))
      with Timer() as eval_timer:
        if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:
          if train_beta:
            self._eval(epoch, metrics, avg_norm_returns, policy_name='beta', log_prefix=True)
          if train_pi:
            self._eval(epoch, metrics, avg_norm_returns, policy_name=policy_name, log_prefix=True)
      
      with Timer() as eval_opex_timer:
        if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:
          if train_beta:
            self._eval_opex(epoch, metrics, avg_norm_returns, policy_name='beta', log_prefix=True)
          if train_pi:
            self._eval_opex(epoch, metrics, avg_norm_returns, policy_name=policy_name, log_prefix=True)

      metrics["train_time"] = train_timer()
      metrics["eval_time"] = eval_timer()
      metrics["eval_opex_time"] = eval_opex_timer()
      metrics["epoch_time"] = train_timer() + eval_timer() + eval_opex_timer()
      self._log(viskit_metrics, metrics)

    # save model
    if self._cfgs.save_model:
      save_data = {"agent": self._agent, #"cfgs": self._cfgs, 
                   "epoch": epoch}
      self._wandb_logger.save_pickle(save_data, "model_final.pkl")

  def _eval(self, epoch, metrics, avg_norm_returns, policy_name="policy", log_prefix=False):
      trajs = self._eval_sampler.sample(
        self._sampler_policy.update_params(
          self._agent.train_params[policy_name]
        ),
        self._cfgs.eval_n_trajs,
        deterministic=True,
        obs_statistics=(self._obs_mean, self._obs_std, self._obs_clip),
      )

      eval_results = {}
      eval_results["eval_average_return"] = np.mean(
        [np.sum(t["rewards"]) for t in trajs]
      )
      eval_results["eval_average_traj_length"] = np.mean(
        [len(t["rewards"]) for t in trajs]
      )
      eval_results["eval_average_normalizd_return"] = np.mean(
        [
          self._eval_sampler.env.get_normalized_score(
            np.sum(t["rewards"])
          ) for t in trajs
        ]
      )
      avg_norm_returns.append(eval_results["eval_average_normalizd_return"])
      eval_results['eval_average_10_normalized_return'] = np.mean(
        avg_norm_returns[-10:]
      )
      eval_results["eval_est_normalized_return"] = max(avg_norm_returns)
      eval_results["eval_done"] = np.mean([np.sum(t["dones"]) for t in trajs])

      if log_prefix:
        metrics.update(prefix_metrics(eval_results, policy_name))
      else:
        metrics.update(eval_results)

      if epoch % 200 == 0:
        if self._cfgs.save_model:
              save_data = {
                "agent": self._agent,
                #"cfgs": self._cfgs,
                "epoch": epoch
              }
              self._wandb_logger.save_pickle(save_data, f"model_eval_{epoch}.pkl")

  @partial(jax.jit, static_argnames=('self', 'num_steps'))
  def opex_action(self, params, observations, action, num_steps):
      """
      Optimize the given action using the Q-function gradient and observations,
      applying gradient ascent for a specified number of steps.
      """
      def q_func(a):
          # Compute the Q-value and take its scalar representation (mean or a single value)
          q_value = self._agent.qf.apply(params['qf1'], observations, a)
          return q_value.mean()  # Ensure the output is a scalar

      optimized_action = action  # Initialize with the original action
      q_value_abs = jnp.abs(self._agent.qf.apply(params['qf1'], observations, action))

      # Perform gradient ascent for the specified number of steps
      for _ in range(num_steps):
          grad_q = jax.grad(q_func)(optimized_action)
          ## divide by q
          if not self.norm_grad : 
            optimized_action = optimized_action + self.opex_beta * grad_q / q_value_abs
          ## divide by \nabla q 
          else : 
            optimized_action = optimized_action + self.opex_beta * grad_q / jnp.abs(grad_q)

          optimized_action = jnp.clip(optimized_action, -1.0, 1.0)

      return optimized_action, action

  def _eval_opex(self, epoch, metrics, avg_norm_returns, policy_name="policy", log_prefix=False):
        """
        Evaluation with OPEX optimization applied, using observations.
        """
        # self.update_opex_beta(epoch)
        
        original_actions = []

        def eval_step(params):
            """
            Define an evaluation step that samples actions and applies OPEX optimization.
            """
            def action_fn(observations, **kwargs):
                # Sample initial action from the policy
                self._sampler_policy.update_params(params[policy_name])

                # Sample initial action from the policy
                action = self._sampler_policy(observations)
                # Optimize the action using OPEX
                opt_a, orig_a = self.opex_action(params, observations, action, self.num_steps)
                original_actions.append(orig_a)
                return opt_a 
            
            return action_fn
        
        # Generate trajectories using the evaluation sampler
        trajs = self._eval_sampler.sample(
            eval_step(self._agent.train_params),
            self._cfgs.eval_n_trajs,
            deterministic=True,
            obs_statistics=(self._obs_mean, self._obs_std, self._obs_clip),
        )

        eval_opex_results = {}
        optimized_actions = np.array([t["actions"] for t in trajs]).reshape(-1)

        eval_opex_results["action_gap"] = np.mean([np.mean(np.linalg.norm(a_1 - a_2, axis = 1) for a_1, a_2 in \
                                                   zip(optimized_actions, original_actions))])
        
        eval_opex_results["opex_beta"] = self.opex_beta

        eval_opex_results["opex_average_return"] = np.mean(
            [np.sum(t["rewards"]) for t in trajs]
        )
        eval_opex_results["opex_average_traj_length"] = np.mean(
            [len(t["rewards"]) for t in trajs]
        )
        eval_opex_results["opex_average_normalizd_return"] = np.mean(
            [
            self._eval_sampler.env.get_normalized_score(
                np.sum(t["rewards"])
            ) for t in trajs
            ]
        )
        avg_norm_returns.append(eval_opex_results["opex_average_normalizd_return"])
        eval_opex_results['opex_average_10_normalized_return'] = np.mean(
            avg_norm_returns[-10:]
        )
        eval_opex_results["opex_best_normalized_return"] = max(avg_norm_returns)
        eval_opex_results["opex_done"] = np.mean([np.sum(t["dones"]) for t in trajs])

        if log_prefix:
            metrics.update(prefix_metrics(eval_opex_results, policy_name))
        else:
            metrics.update(eval_opex_results)

        if epoch % 200 == 0:
          if self._cfgs.save_model:
                  save_data = {
                  "agent": self._agent,
                  #"cfgs": self._cfgs,
                  "epoch": epoch
                  }

                  self._wandb_logger.save_pickle(save_data, f"model_eval_opex_{epoch}.pkl")

    
  # def _log(self, viskit_metrics, metrics):
  #   # train 결과 로깅
  #   self._wandb_logger.log({"train_metrics": {k: v for k, v in metrics.items() if k.startswith("train_")}})
  #   # eval 결과 로깅
  #   self._wandb_logger.log({"eval_metrics": {k: v for k, v in metrics.items() if k.startswith("eval_")}})
  #   # eval_opex 결과 로깅
  #   self._wandb_logger.log({"eval_opex_metrics": {k: v for k, v in metrics.items() if k.startswith("opex_")}})
  #   self._wandb_logger.log({
  #       "beta_info": {
  #           "opex_beta": self.opex_beta,
  #       }
  #   })

  #   viskit_metrics.update(metrics)
  #   logger.record_dict(viskit_metrics)
  #   logger.dump_tabular(with_prefix=False, with_timestamp=False)
  def _log(self, viskit_metrics, metrics):
    self._wandb_logger.log(metrics)
    viskit_metrics.update(metrics)
    logger.record_dict(viskit_metrics)
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

  def _setup(self):
    set_random_seed(self._cfgs.seed)

    # setup logger
    self._wandb_logger = self._setup_logger()

    # setup dataset and eval_sample
    self._dataset, self._eval_sampler = self._setup_dataset()

    # setup policy
    print("test mode ", self.test_mode)
    if not self.test_mode : 
      self._policy = self._setup_policy()

      # setup Q-function
      self._qf = self._setup_qf()

      # setup vf only for IQL
      if self._algo_type in [ALGO.IQL, ALGO.IQL_Onestep]:
        self._vf = self._setup_vf()
    else :
      # load policy 
      loaded_agent = self._wandb_logger.load_pickle(self._cfgs.logging.model_dir)
      self._policy = loaded_agent.policy

      # setup Q-function
      self._qf = loaded_agent.qf

      # setup vf only for IQL
      if self._algo_type in [ALGO.IQL, ALGO.IQL_Onestep]:
        self._vf = loaded_agent.vf

    # setup agent
    max_steps = int(self._training_cfgs.n_epochs * self._cfgs.n_train_step_per_epoch)
    if self._algo_type in [ALGO.Onestep, ALGO.IQL_Onestep]:
      max_steps = int(self._training_cfgs.pi_n_epochs * self._cfgs.n_train_step_per_epoch)
    if self._algo_type in [ALGO.IQL, ALGO.IQL_Onestep]:
      self._agent = self._algo(
        self._agent_cfgs,
        self._policy,
        self._qf,
        self._vf,
        max_steps=max_steps
      )
    else:
      self._agent = self._algo(
        self._agent_cfgs, self._policy, self._qf, max_steps=max_steps
      )


    # setup sampler policy
    self._sampler_policy = SamplerPolicy(
      self._agent.policy, self._agent.train_params["policy"]
    )

  def _setup_logger(self):
    env_name_high = ENVNAME_MAP[self._env]
    env_name_full = self._cfgs.env
    dataset_name_abbr = DATASET_ABBR_MAP[self._cfgs.dataset]

    logging_configs = self._cfgs.logging
    if self._cfgs.release:
      logging_configs["project"] = f"{env_name_high}-{dataset_name_abbr}"
    else:
      # logging_configs[
      #   "project"
      # ] = f"{self._cfgs.algo_cfg.name}-{env_name_high}-{dataset_name_abbr}"
      logging_configs["project"] = f""

    if self._training_cfgs.note != '':
      logging_configs["project"] += f"-{self._training_cfgs.note}"

    wandb_logger = WandBLogger(
      config=logging_configs, variant=self._variant, env_name=env_name_full
    )
    setup_logger(
      variant=self._variant,
      exp_id=wandb_logger.experiment_id,
      seed=self._cfgs.seed,
      base_log_dir=self._cfgs.logging.output_dir,
      include_exp_prefix_sub_dir=False,
    )

    return wandb_logger

  def _setup_d4rl(self):
    eval_sampler = TrajSampler(
      gym.make(self._cfgs.env), self._cfgs.max_traj_length
    )

    norm_reward = self._training_cfgs.norm_reward
    if self._env == ENV.Antmaze:
      norm_reward = False

    rew_clip_val = None
    if self._env == ENV.Adroit and self._training_cfgs.rew_clip:
      rew_clip_val = ENV_REW_CLIP_VAL.get(self._cfgs.env.split('-')[0], None)
    dataset = get_d4rl_dataset(
      self._env,
      eval_sampler.env,
      self._agent_cfgs.nstep,
      self._agent_cfgs.discount,
      top_n=self._cfgs.topn,
      norm_reward=norm_reward,
      rew_clip_val=rew_clip_val,
    )

    dataset["rewards"] = dataset[
      "rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
    
    dataset["actions"] = np.clip(
      dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
    )
    dataset["next_actions"] = np.concatenate([dataset["actions"][1:], np.zeros_like(dataset["actions"][0:1])])

    if self._env == ENV.Kitchen or self._env == ENV.Adroit or self._env == ENV.Antmaze:
      if self._training_cfgs.obs_norm:
        self._obs_mean = dataset["observations"].mean()
        self._obs_std = dataset["observations"].std()
        self._obs_clip = 10
      norm_obs(dataset, self._obs_mean, self._obs_std, self._obs_clip)

      if self._env == ENV.Antmaze:
        if self._algo_type in [ALGO.IQL, ALGO.IQL_Onestep]:
          dataset["rewards"] -= 1
        else:
          dataset["rewards"] = (dataset["rewards"] - 0.5) * 4
      else:
        min_r, max_r = np.min(dataset["rewards"]), np.max(dataset["rewards"])
        dataset["rewards"] = (dataset["rewards"] - min_r) / (max_r - min_r)
        dataset["rewards"] = (dataset["rewards"] - 0.5) * 2

    # set sampler
    dataset = Dataset(dataset)
    sampler = RandSampler(dataset.size(), self._training_cfgs.batch_size)
    dataset.set_sampler(sampler)

    return dataset, eval_sampler

  def _setup_rlup(self):
    path = Path(__file__).absolute().parent.parent / 'rlup_data'
    if self._algo_type == ALGO.BC:
      dataset = BC_RLUPDataset(
        self._cfgs.rl_unplugged_task_class,
        self._cfgs.env,
        batch_size=self._training_cfgs.batch_size,
        action_clipping=self._cfgs.clip_action,
        top_n=self._cfgs.topn,
      )
    else:
      assert self._cfgs.topn==100, "Percentile algos are not implemented except BC."
      dataset = RLUPDataset(
        self._cfgs.rl_unplugged_task_class,
        self._cfgs.env,
        str(path),
        batch_size=self._training_cfgs.batch_size,
        action_clipping=self._cfgs.clip_action,
      )

    env = DM2Gym(dataset.env)
    eval_sampler = TrajSampler(env, max_traj_length=self._cfgs.max_traj_length)

    return dataset, eval_sampler

  def _setup_dataset(self):
    self._obs_mean = 0
    self._obs_std = 1
    self._obs_clip = np.inf

    dataset_type = DATASET_MAP[self._cfgs.dataset]

    if dataset_type == DATASET.D4RL:
      dataset, eval_sampler = self._setup_d4rl()
    elif dataset_type == DATASET.RLUP:
      dataset, eval_sampler = self._setup_rlup()
    else:
      raise NotImplementedError

    self._observation_dim = eval_sampler.env.observation_space.shape[0]
    self._action_dim = eval_sampler.env.action_space.shape[0]
    self._max_action = float(eval_sampler.env.action_space.high[0])

    if self._agent_cfgs.target_entropy >= 0.0:
      action_space = eval_sampler.env.action_space
      self._agent_cfgs.target_entropy = -np.prod(action_space.shape).item()

    return dataset, eval_sampler

  def _setup_policy(self):
    if self._algo_type in [
      ALGO.CRR, ALGO.IQL, ALGO.CQL, ALGO.SAC, ALGO.BC, ALGO.Onestep, ALGO.IQL_Onestep,
    ]:
      log_sig_min, log_sig_max = None, None
      use_log_std_multiplier = True
      if self._algo_type in [ALGO.IQL, ALGO.IQL_Onestep]:
        log_sig_min, log_sig_max = -5, 2
        use_log_std_multiplier = False
      if self._algo_type == ALGO.Onestep:
        log_sig_min, log_sig_max = -5, 0
        use_log_std_multiplier = False
      policy = TanhGaussianPolicy(
        self._observation_dim,
        self._action_dim,
        self._training_cfgs.policy_arch,
        self._training_cfgs.orthogonal_init,
        self._cfgs.policy_log_std_multiplier,
        self._cfgs.policy_log_std_offset,
        use_layer_norm=self._training_cfgs.policy_layer_norm,
        state_dependent_std=self._training_cfgs.state_dependent_std,
        tanh_squash_distribution=self._training_cfgs.tanh_squash_distribution,
        clip_mean=self._training_cfgs.clip_mean,
        log_sig_min=log_sig_min,
        log_sig_max=log_sig_max,
        use_log_std_multiplier=use_log_std_multiplier,
        dropout=self._training_cfgs.dropout,
        last_layer_init=self._training_cfgs.last_layer_init,
      )
      if self._training_cfgs.use_resnet:
        policy = ResTanhGaussianPolicy(
          self._observation_dim,
          self._action_dim,
          self._cfgs.policy_log_std_multiplier,
          self._cfgs.policy_log_std_offset,
          state_dependent_std=self._cfgs.state_dependent_std,
          tanh_squash_distribution=self._cfgs.tanh_squash_distribution,
          log_sig_min=log_sig_min,
          log_sig_max=log_sig_max,
          use_log_std_multiplier=use_log_std_multiplier,
          hidden_dim=self._training_cfgs.hidden_dim,
          num_blocks=self._training_cfgs.num_blocks,
          res_type=self._training_cfgs.res_type
        )
    elif self._algo_type == ALGO.TD3:
      policy = DirectMappingPolicy(
        self._observation_dim,
        self._action_dim,
        self._max_action,
        self._training_cfgs.policy_arch,
        self._training_cfgs.orthogonal_init,
        dropout=self._training_cfgs.dropout,
        last_layer_init=self._training_cfgs.last_layer_init,
      )
      if self._training_cfgs.use_resnet:
        policy = ResDirectMappingPolicy(
          self._observation_dim,
          self._action_dim,
          self._max_action,
          hidden_dim=self._training_cfgs.hidden_dim,
          num_blocks=self._training_cfgs.num_blocks,
          res_type=self._training_cfgs.res_type
        )
    else:
      raise NotImplementedError

    return policy

  def _setup_qf(self):
    if self._algo_type in [
      ALGO.CRR,
      ALGO.IQL,
      ALGO.CQL,
      ALGO.TD3,
      ALGO.SAC,
      ALGO.BC,
      ALGO.Onestep,
      ALGO.IQL_Onestep,
      
    ]:
      qf = FullyConnectedQFunction(
        self._observation_dim,
        self._action_dim,
        self._training_cfgs.qf_arch,
        self._training_cfgs.orthogonal_init,
        self._training_cfgs.qf_layer_norm,
        self._training_cfgs.activation,
        last_layer_init=self._training_cfgs.last_layer_init,
      )
      if self._training_cfgs.use_resnet:
        qf = ResQFunction(
          self._observation_dim,
          self._action_dim,
          self._training_cfgs.hidden_dim,
          self._training_cfgs.num_blocks,
          self._training_cfgs.res_type,
        )
    else:
      raise NotImplementedError

    return qf

  def _setup_vf(self):
    vf = FullyConnectedVFunction(
      self._observation_dim,
      self._training_cfgs.qf_arch,
      self._training_cfgs.orthogonal_init,
      self._training_cfgs.qf_layer_norm,
      self._training_cfgs.activation,
      last_layer_init=self._training_cfgs.last_layer_init,
    )
    if self._training_cfgs.use_resnet:
      vf = ResVFunction(
        self._observation_dim,
        self._training_cfgs.hidden_dim,
        self._training_cfgs.num_blocks,
        self._training_cfgs.res_type,
      )
    return vf