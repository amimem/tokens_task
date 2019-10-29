import gym
import gym_tokens
import argparse
import random

import time
import datetime
import sys
import utils
import lib

import numpy as np


numNT = 31
numHT = 31

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--games", type=int, default=500, help="number of games training (default: 500)")
	parser.add_argument("--env",  default='tokens-v0' , help="name of the environment to train on (REQUIRED)")
	parser.add_argument("--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
	parser.add_argument("--seed", type=int, default=7, help="random seed (default: 7)")
	parser.add_argument("--log_interval", type=int, default=1, help="number of updates between two logs (default: 1)")
	parser.add_argument("--algo", default='q-learning', help="algorithm to use: SARSA | Q-learning")
	parser.add_argument("--convg", type=float, default=0.01, help="convergence value")
	parser.add_argument("--frames", type=int, default=15*1000, help="number of frames of training (default: 3000)")
	parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
	parser.add_argument("--save-interval", type=int, default=30, help="number of updates between two saves (default: 30, 0 means no saving)")


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
	txt_logger.info("Environments loaded\n")

	# Load training status

	# try:
	# 	status = utils.get_status(model_dir)
	# except OSError:
	# 	status = {"num_frames": 0, "update": 0, "num_games":0}
	status = {"num_frames": 0, "update": 0, "num_games":0}
	txt_logger.info("Training status loaded\n")

	num_states = env.get_num_states()
	num_actions = env.get_num_actions()

	model = lib.Q_Table(env.get_num_states(), env.get_num_actions(), (numNT, numHT), args.convg)
	policy = lib.EpsilonGreedyPolicy()
	eps_track = lib.EpsilonTracker(1.0, 0.01, 15*200, policy)
	monkeyAgent = lib.SarsaAgent(policy, model)

	num_frames = status["num_frames"]
	update = status["update"]
	num_games = status["num_games"]

	start_time = time.time()
	totalReturns = [] 
	totalLoss = []

	state = env.reset()


	while num_frames < args.frames: 

		# print('state: ', [state])

		eps_track.set_eps(num_frames)
		action = monkeyAgent.get_actions([state])
		next_state, reward, is_done, _ = env.step(action)
		next_act = monkeyAgent.get_actions([next_state])

		loss = model.get_TDerror([state], action, [next_state], next_act, reward)
		model.update_qVal(args.lr, [state], action, loss)
		totalLoss.append(loss)

		state = next_state

		num_frames+=1 
		update+= 1

		if is_done:
			num_games+=1
			totalReturns.append(reward)

			# print('Episode over!')
			# print('state: ', [state])


		if num_games > 0 and num_games % args.log_interval == 0:

			duration = int(time.time() - start_time)
			totalLoss_val = np.sum(totalLoss)
			totalReturn_val = np.sum(totalReturns)

			avg_loss = totalLoss_val/num_frames
			avg_returns = totalReturn_val / num_games

			header = ["update", "frames", "Games", "duration"]
			data = [update, num_frames, num_games, duration]

			header += ["eps"]
			data += [policy.epsilon]

			header += ["Loss", "Returns", "Avg Loss", "Avg Returns"]
			data += [totalLoss_val.item(), totalReturn_val.item(), avg_loss.item(), avg_returns.item()]

			txt_logger.info(
				"U {} | F {} | G {} | D {} | EPS {:.3f} | L {:.3f} | R {:.3f} | Avg L {:.3f} | Avg R {:.3f}"
				.format(*data))

			# header += ["Loss", "Returns", "Avg Loss", "Avg Returns"]
			# data += [totalLoss_val, totalReturn_val, avg_loss, avg_returns]


			# txt_logger.info(
			# 	"U {} | F {} | G {} | D {} | EPS {:.3f} | L {:.3f}"
			# 	.format(*data))

			if status["num_frames"] == 0:
				csv_logger.writerow(header)
			csv_logger.writerow(data)
			csv_file.flush()

		# Save status
		# if args.save_interval > 0 and update % args.save_interval == 0:
		# 	status = {"num_frames": num_frames, "update": update, "games": num_games}
		# 	model.save_q_state(model_dir)
		# 	txt_logger.info("Status saved")


	# start_state = env.reset()

	# traj_states, traj_actions, traj_rewards, traj_isDone  = [], [], [], []

	# traj_states.append(start_state)

	# print(start_state)
	# print(start_state.shape)

	# # step = random.randint(0,15)
	# # choice = random.choice([0,1,2])

	# print(monkeyAgent.get_actions([start_state]))


	# for i in range(1):

	# 	state = env.reset()

	# 	for game_step in range(15):

	# 		eps_track.set_eps(game_step)

	# 		action = monkeyAgent.get_actions([state])

	# 		print('action: ', action)

	# 		next_state, reward, is_done, _ = env.step(action)

	# 		traj_states.append(next_state.tolist())
	# 		traj_actions.append(action)
	# 		traj_rewards.append(reward)
	# 		traj_isDone.append(is_done)

	# 		state = next_state


	# 		if is_done:
	# 			break


		# step = random.randint(0,15)
		# choice = random.choice([-1,0,1])

		# for game_step in range(15):

		# 	if game_step == step:
		# 		action = choice

		# 	else:
		# 		action = 0

		# 	next_state, reward, is_done, _ = env.step(action)
		# 	traj_states.append(next_state.tolist())
		# 	traj_actions.append(action)
		# 	traj_rewards.append(reward)
		# 	traj_isDone.append(is_done)

		# 	if is_done:
		# 		break

		# print()
		# print('step: ', step)
		# print('choice: ', choice)
		# print('state: ', traj_states)
		# print('actions: ', traj_actions)
		# print('reward: ', traj_rewards)
		# print('is_done: ', traj_isDone)

if __name__ == '__main__':
	main()