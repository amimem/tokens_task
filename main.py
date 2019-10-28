import gym
import gym_tokens
import argparse
import random

import time
import datetime
import sys
import utils


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--games", type=int, default=500, help="number of games training (default: 500)")
	parser.add_argument("--env",  default='tokens-v0' , help="name of the environment to train on (REQUIRED)")
	parser.add_argument("--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
	parser.add_argument("--seed", type=int, default=7, help="random seed (default: 7)")
	parser.add_argument("--log-interval", type=int, default=1, help="number of updates between two logs (default: 1)")
	parser.add_argument("--algo", default='q-learning', help="algorithm to use: a2c | ppo (REQUIRED)")

	args = parser.parse_args()

	#create train dir
	date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

	model_name = args.model or default_model_name
	model_dir = utils.get_model_dir(model_name)

	# Load loggers and Tensorboard writer

	txt_logger = utils.get_txt_logger(model_dir)
	csv_file, csv_logger = utils.get_csv_logger(model_dir)

	# Log command and all script arguments

	txt_logger.info("{}\n".format(" ".join(sys.argv)))
	txt_logger.info("{}\n".format(args))

	# Set seed for all randomness sources
	utils.seed(args.seed)

	env = gym.make('tokens-v0')

	start = env.reset()

	traj_states, traj_actions, traj_rewards, traj_isDone  = [], [], [], []

	traj_states.append(start)

	# step = random.randint(0,15)
	# choice = random.choice([0,1,2])


	for i in range(1):

		step = random.randint(0,15)
		choice = random.choice([-1,0,1])

		for game_step in range(15):

			if game_step == step:
				action = choice

			else:
				action = 0

			next_state, reward, is_done, _ = env.step(action)
			traj_states.append(next_state.tolist())
			traj_actions.append(action)
			traj_rewards.append(reward)
			traj_isDone.append(is_done)

			if is_done:
				break

		print()
		print('step: ', step)
		print('choice: ', choice)
		print('state: ', traj_states)
		print('actions: ', traj_actions)
		print('reward: ', traj_rewards)
		print('is_done: ', traj_isDone)



if __name__ == '__main__':
	main()