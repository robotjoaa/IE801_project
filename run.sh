#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

export XLA_PYTHON_CLIENT_MEM_FRACTION=.40

python -m experiments.main \
  --env "antmaze-large-diverse-v2" \
  --logging.output_dir="./experiment_output" \
  --algo_cfg="./configs/iql_cfgs.py" \
  --save_model True \
  --logging.online 
  #   --logging.model_dir="./experiment_output/93692b6c24324cd8bf27261a51b9e966/model_0.pkl" \