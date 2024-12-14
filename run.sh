#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.40

python -m experiments.main \
  --env "antmaze-medium-diverse-v2" \
  --logging.output_dir="./experiment_output" \
  --algo_cfg="./configs/iql_cfgs.py" \
  --logging.model_dir="./experiment_output/93692b6c24324cd8bf27261a51b9e966/model_0.pkl" \
  --save_model True \
  #--logging.online \
  