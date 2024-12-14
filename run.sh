#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.40

python -m experiments.main \
  --env "antmaze-medium-diverse-v2" \
  --logging.output_dir="./experiment_output" \
  --algo_cfg="./configs/iql_cfgs.py" \
  --logging.online \
  --save_model True