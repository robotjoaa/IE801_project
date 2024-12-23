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

import ml_collections
import algos
from configs.base_cfgs import get_default_training_cfgs
import numpy as np

def get_training_configs():
  training_cfgs = get_default_training_cfgs()
  config = ml_collections.ConfigDict()

  config.batch_size = 256
  config.activation = "relu"
  config.obs_norm = False
  config.state_dependent_std = False
  config.tanh_squash_distribution = False
  config.norm_reward = True
  config.clip_mean = False
  config.orthogonal_init = True
  config.last_layer_init = np.sqrt(2.0)

  training_cfgs.update(config)

  return training_cfgs 

def get_config():
  config = ml_collections.ConfigDict()

  config.name = "IQL"
  config.type = 'model-free'
  config.training = get_training_configs()
  config.agent = getattr(algos, config.name).get_default_config()
  
  return config
