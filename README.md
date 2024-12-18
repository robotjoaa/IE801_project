# IE801 Project (Team 11)

Team 11 : JeongWoo Park (20243347), Sojeong Rhee (20243606)

Title : Test time adaptation in Offline RL

This repository is forked from [Offbench](https://github.com/sail-sg/offbench)

## Main Idea
The project is based on the [OPEX](https://arxiv.org/pdf/2406.09329), introduced in Park et al., Is Value Learning Really the Main Bottleneck in Offline RL? (NeurIPS 2024 Workshop) 
![image](https://github.com/user-attachments/assets/9f3579a5-6894-4c70-a3af-29331228f3e2)
Since the paper only reported single step IQL results without implementation details, we applied multi-step OPEX and normalization with gradient norm settings.

## Methods
### Hyperparameter Search
![image](https://github.com/user-attachments/assets/c1210283-1b21-46d8-a48f-3f4f13f688c2)

## Setup the Environment

To setup the environment, we recommend to use docker. Simply run
```bash
./docker_run.sh
```

## Run Experiments
Inside docker container, simply run
```bash
./run.sh
```
You can modify run.sh file with specific environments and algorithms.

## Weights and Biases Online Visualization Integration 
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable and add `--logging.online` when launching the script.
Alternatively, you could simply run `wandb login`.

## Results
### Overall Results on Antmaze dataset
![image](https://github.com/user-attachments/assets/977c7380-47ad-4397-8821-83c051fcda2c)
![image](https://github.com/user-attachments/assets/3a10c248-eb07-4c4d-af12-b3bf724df242)

### Results on Antmaze-Umaze-Diverse-v2, num_steps = 1
![image](https://github.com/user-attachments/assets/c763b5f9-f135-47aa-a197-220bfa24d513)
![image](https://github.com/user-attachments/assets/66caa97b-d588-42fa-94c6-26a996a87621)
![image](https://github.com/user-attachments/assets/711fe2c6-9cb1-4397-8816-4a36581193dc)