# [Tokens Task: An Empirical Study](https://github.com/user-attachments/files/16913932/TokensProject_AminM.pdf)

This paper explores the performance of reinforcement learning (RL) agents on the "tokens task," a decision-making test commonly used in neuroscience. The study compares the behavior of these RL agents with primates, demonstrating that RL agents, when provided with a fully observable environment and optimized hyperparameters, consistently find the optimal solution. Various algorithms, such as Q-learning and SARSA, are tested, with results showing that exploration methods like Boltzmann improve performance. The findings suggest RL agents can effectively mimic decision-making processes under certain conditions, albeit with limitations in real-world comparisons.

## Environment Installation

To install the Gym environment, go to the local directory of this folder: 

```bash
python setup.py install
```

## Results Reproduction

We used Conda environments to run the experiments, to create the same environment run the following commands:

```bash
conda create --name <env> --file conda-requirements.txt
pip install -r pip-requirements.txt
```

After installing the requirements run the following commands:

```bash
python main.py --games 10000 --env tokens-v0 --variation horizon --algo q-learning --lr 1.0 --lr_final 0.001 --seed 0 --height 11 --gamma 0.8 --fancy_discount --eps_start 0.01 --eps_final 0.0001 --eps_games 10000

python main.py --games 10000 --env tokens-v0 --variation horizon --algo q-learning --lr 1.0 --lr_final 0.001 --seed 0 --height 11 --gamma 0.8 --fancy_discount --eps_soft --eps_start 0.01 --eps_final 0.0001 --eps_games 100[TokensProject_AminM.pdf](https://github.com/user-attachments/files/16913931/TokensProject_AminM.pdf)
00[TokensProject_AminM.pdf](https://github.com/user-attachments/files/16913930/TokensProject_AminM.pdf)


python main.py --games 10000 --env tokens-v0 --variation horizon --algo q-learning --lr 1.0 --lr_final 0.001 --seed 0 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo e-sarsa --lr 1.0 --lr_final 0.001 --seed 0 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo e-sarsa --lr 1.0 --lr_final 0.001 --seed 10 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo e-sarsa --lr 1.0 --lr_final 0.001 --seed 20 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo q-learning --lr 1.0 --lr_final 0.001 --seed 0 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo q-learning --lr 1.0 --lr_final 0.001 --seed 10 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo q-learning --lr 1.0 --lr_final 0.001 --seed 20 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo sarsa --lr 1.0 --lr_final 0.001 --seed 0 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo sarsa --lr 1.0 --lr_final 0.001 --seed 10 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python main.py --games 10000 --env tokens-v0 --variation terminate --algo sarsa --lr 1.0 --lr_final 0.001 --seed 20 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax

python reinforce2.py --games 10000 --env tokens-v0 --variation terminate --seed 0 --height 11 --gamma 0.99 --fancy_discount

python reinforce2.py --games 50000 --env tokens-v0 --variation terminate --seed 0 --height 11 --gamma 0.99 --fancy_discount
```



To see how the agent performs on the POMDP run the commands with `--env tokens-v1 `, for example:

```bash
main.py --games 10000 --env tokens-v1 --variation terminate --algo sarsa --lr 1.0 --lr_final 0.001 --seed 0 --height 11 --gamma 0.8 --fancy_discount --tmp_start 0.01 --tmp_final 0.0001 --tmp_games 10000 --softmax
```

Each command creates a subdirectory in the storage directory, to create the graphs copy the path of the subdirectory into the Jupyter notebook in notebooks/tokens_task_analysis_RL_onerun.ipynb, do not forget to adjust `T` in the notebook to the specified `--height` in the commands.

## Youtube Summary Video

[Click here to see the video](https://youtu.be/lOqUZJVFAzg)
