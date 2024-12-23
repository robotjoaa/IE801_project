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

import os

import absl.app

# from experiments.mf_trainer import MFTrainer
from experiments.mf_trainer_opex import MFTrainer_OPEX
# from experiments.mf_trainer_opex_2 import MFTrainer_OPEX
import d4rl
import argparse


FLAGS = absl.flags.FLAGS

def main(argv):
  model_type = absl.flags.FLAGS.algo_cfg.type
  if absl.flags.FLAGS.release:
    assert not os.path.isdir("./.git")


  if model_type == "model-free":
    # mf_trainer = MFTrainer()
    mf_trainer = MFTrainer_OPEX()

    mf_trainer.train()
  else:
    raise NotImplementedError


if __name__ == '__main__':
  absl.app.run(main)
