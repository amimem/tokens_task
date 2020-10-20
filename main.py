import gym
import gym_tokens
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from  torch.distributions import Categorical
import scipy.signal as signal

import time
import datetime
import sys
import utils
import lib

import numpy as np

num_actions = 3

def _sign(num):

	if num < 0:
		return -1

	elif num > 0:
		return 1

	else:
		return 0

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--games", type=int, default=100000, help="number of games training (default: 500)")
	parser.add_argument("--env",  default='tokens-v0' , help="name of the environment to train on (REQUIRED)")
	parser.add_argument("--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
	parser.add_argument("--seed", type=int, default=7, help="random seed (default: 7)")
	parser.add_argument("--log_interval", type=int, default=1, help="number of updates between two logs (default: 1)")
	parser.add_argument("--algo", default='sarsa', help="algorithm to use: sarsa | q-learning | e-sarsa | double-q")
	parser.add_argument("--convg", type=float, default=0.00001, help="convergence value")
	parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
	parser.add_argument("--lr_final", type=float, default=0.0001, help="learning rate")
	parser.add_argument("--save-interval", type=int, default=1000, help="number of updates between two saves (default: 30, 0 means no saving)")
	parser.add_argument("--eps_start", type=float, default=1.0, help="initial epsilon-greedy value")
	parser.add_argument("--eps_final", type=float, default=0.01, help="final epsilon-greedy value")
	parser.add_argument("--eps_games", type=int, default=40000, help="number of frames for eps greedy to go from init value to final value (default: 75k)")
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
	parser.add_argument("--height", type=int, default=15, help="game tree height")
	parser.add_argument('--fancy_discount', help='use fancy discounting rewards',action='store_true')
	parser.add_argument('--fast_block', help='fast block discounting',action='store_true')
	parser.add_argument('--fancy_eps', help='try to do epsilon-greedy per game rather than per step',action='store_true')
	parser.add_argument('--fancy_tmp', help='try to do softmax per game rather than per step',action='store_true')
	parser.add_argument("--tmp_start", type=float, default=1.0, help="initial temperature value")
	parser.add_argument("--tmp_final", type=float, default=0.01, help="final temperature value")
	parser.add_argument("--tmp_games", type=int, default=10000, help="number of frames for temperature to go from init value to final value (default: 75k)")
	parser.add_argument('--softmax', help='use softmax exploration',action='store_true')
	parser.add_argument('--eps_soft', help='use epsilon soft exploration',action='store_true')
	parser.add_argument('--variation', default="horizon", help='which variation')
	parser.add_argument('--reward', type=int, default="1", help='fixed reward')


	args = parser.parse_args()

	#create train dir
	date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	default_model_name = f"{args.env}_{args.variation}_{args.fancy_discount}_{args.algo}_seed{args.seed}_{date}"

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

	if args.fast_block:
		block_discount = 0.25

	else:
		block_discount = 0.75

	env = gym.make(args.env, alpha=block_discount, seed=args.seed, terminal=args.height, fancy_discount=args.fancy_discount, v=args.variation)
	txt_logger.info("Environments loaded\n")

	return_zero = False

	# Load training status

	# try:
	# 	status = utils.get_status(model_dir)
	# except OSError:
	# 	status = {"num_frames": 0, "update": 0, "num_games":0}
	status = {"num_frames": 0, "update": 0, "num_games":0}
	txt_logger.info("Training status loaded\n")

	num_states = env.get_num_states()
	num_actions = env.get_num_actions()
	total_run_time_steps = args.height * args.games # Total time-steps

	numNT = (args.height * 2) + 1  # -15 to 15
	numHT = (args.height * 2) + 1 # -15 to 15

	# model = lib.Q_Table(env.get_num_states(), env.get_num_actions(), (numNT, numHT), args.convg)
	if env.observation_space.shape[0] == 3:
		if args.algo == 'double-q':
			model = lib.Q_Table(numNT*numHT*(args.height+1), num_actions, (numNT, numHT, args.height), args.convg, args.height) 
			model2 = lib.Q_Table(numNT*numHT*(args.height+1), num_actions, (numNT, numHT, args.height), args.convg, args.height)
		else:
			model = lib.Q_Table(numNT*numHT*(args.height+1), num_actions, (numNT, numHT, args.height), args.convg, args.height) 
	else:
		if args.algo == 'double-q':
			model = lib.Q_Table(numNT*numHT, num_actions, (numNT, numHT), args.convg, args.height) 
			model2 = lib.Q_Table(numNT*numHT, num_actions, (numNT, numHT), args.convg, args.height)
		else:
			model = lib.Q_Table(numNT*numHT, num_actions, (numNT, numHT), args.convg, args.height) 


	#NOTE why the number of states is the way it is ? num_states x (height + 2)

	if args.softmax:
		policy = lib.SoftmaxPolicy()
		if args.fancy_tmp:
			tmp_track = lib.TemperatureTracker(args.tmp_start, args.tmp_final, args.tmp_games, policy) # tmp is changed from game to game
		else:
			tmp_track = lib.TemperatureTracker(args.tmp_start, args.tmp_final, args.tmp_games*args.height, policy)

	elif args.eps_soft:
		policy = lib.EpsilonSoftPolicy()
		if args.fancy_eps:
			eps_track = lib.EpsilonTracker(args.eps_start,args.eps_final, args.eps_games, policy)
		else:
			eps_track = lib.EpsilonTracker(args.eps_start,args.eps_final, args.eps_games*args.height, policy)

	elif args.fancy_eps:
		#TODO
		policy = lib.EpsilonGreedyGamePolicy()
		eps_track = lib.EpsilonTracker(args.eps_start,args.eps_final, args.eps_games, policy)

	else:
		policy = lib.EpsilonGreedyPolicy(epsilon=args.eps_start)
		eps_track = lib.EpsilonTracker(args.eps_start,args.eps_final, args.eps_games*args.height, policy) # args.eps_games*args.height is the number of total time_step for decreasing epsilon

	if args.algo == 'sarsa': 
		monkeyAgent = lib.SarsaAgent(policy, model, args.height)

	elif args.algo == 'q-learning':
		monkeyAgent = lib.QlAgent(policy, model, args.height)

	elif args.algo == 'e-sarsa':
		monkeyAgent = lib.ExpectedSARSA(policy, model, args.height)

	elif args.algo == 'double-q':
		monkeyAgent = lib.DoubleQLearning(policy, model, model2, args.height)

	lr_sched = lib.LRscheduler(args.lr, args.lr_final, total_run_time_steps) 
	#NOTE is there is reason that lr is not decreased to the final value during the experiment?

	num_frames = status["num_frames"]
	update = status["update"]
	num_games = status["num_games"]
	num_games_prevs = 0

	start_time = time.time()
	totalReturns = [] # Return per episode
	totalLoss = [] # loss trajectory

	env.set_reward(args.reward)

	state, game_time_step  = env.reset()
	decisionTime = []
	lossPerEpisode = []
	train_info = []

	last_choice = 0

	decisionTime = np.zeros(shape=((args.height*2)+1)) # histogram of decision times per episode
	#NOTE why first 15 elements are always empty

	# info = []

	traj_group = []

	choice_made = []
	correct_choice = []
	finalDecisionTime = []
	finalRewardPerGame = []
	numCorrectChoice = 0
	numRecentCorrectChoice = []

	took_action = False

	while num_games < args.games: 

		if args.softmax:
			if args.fancy_tmp:
				tmp_track.set_tmp(num_games)
			else:
				tmp_track.set_tmp(num_frames)
			# decrease temperature from game to game
			#NOTE change per episode or per time_step?

		elif args.eps_soft:
			#NOTE change per episode or per time_step?
			if args.fancy_eps:
				eps_track.set_eps(num_games) # decrease epsilon from game to game
			else:
				eps_track.set_eps(num_frames)

		elif args.fancy_eps:
			eps_track.set_eps(num_games) # epsilon in fancy eps is changed from game to game

		else:
			eps_track.set_eps(num_frames) # otherwise it is changes timestep to timestep

		# env.render()
		# time.sleep(0.25)

		if env.v == 'horizon' and return_zero:
			if not took_action:
				action = monkeyAgent.get_actions(state, False, game_time_step)
				next_state, reward, is_done, game_time_step = env.step(action)
				if action:
					took_action = True
			else:
				action = monkeyAgent.get_actions(state, False, game_time_step, True)
				next_state, reward, is_done, game_time_step = env.step(action)

		else: # if not in the horizon setting
				action = monkeyAgent.get_actions(state, False, game_time_step)
				next_state, reward, is_done, game_time_step = env.step(action)



		lr = lr_sched.get_lr(num_frames) # learning rate is changed from timestep to timestep
		
		if args.algo == 'sarsa':
			next_act = monkeyAgent.get_actions(next_state, False, game_time_step)
			loss = model.get_TDerror(state, action, next_state, next_act, reward, args.gamma, is_done, args.algo)
			converged = model.update_qVal(lr, state, action, loss)
		elif args.algo == 'e-sarsa':
			next_act, probs = monkeyAgent.get_actions(next_state, True, game_time_step)
			loss = model.get_TDerror(state, action, next_state, probs, reward, args.gamma, is_done, args.algo)
			converged = model.update_qVal(lr, state, action, loss)
		else:
			next_act = None
			if args.algo == 'double-q':
				if np.random.binomial(1,0.5):
					loss = model.get_TDerror(state, action, next_state, next_act, reward, args.gamma, is_done, args.algo, model2)
					converged = model.update_qVal(lr, state, action, loss)
				else:
					loss2 = model2.get_TDerror(state, action, next_state, next_act, reward, args.gamma, is_done, args.algo, model)
					converged = model2.update_qVal(lr, state, action, loss2)
			else: # for q-learning
				loss = model.get_TDerror(state, action, next_state, next_act, reward, args.gamma, is_done, args.algo)
				converged = model.update_qVal(lr, state, action, loss)

		totalLoss.append(loss) # loss trajectory

		if is_done:
			env.close()
			num_games+=1

			if args.env == 'tokens-v3' or args.env == 'tokens-v4':
				totalReturns.append(env.reward) # reward per episode
				if env.reward > 0:
					numCorrectChoice += 1
					numRecentCorrectChoice.append(1)
				else:
					numRecentCorrectChoice.append(0) # binary value, correct choice or not per episode

			else:
				totalReturns.append(reward) # reward per episode
				if reward > 0:
					numCorrectChoice += 1
					numRecentCorrectChoice.append(1)
				else:
					numRecentCorrectChoice.append(0) # binary value, correct choice or not per episode

			lossPerEpisode.append(np.sum(totalLoss))
			totalLoss = []

			decision_step = model._augState(abs(next_state[1])) # taking abs means that decision step is always between 15 and 31
			decisionTime[decision_step-1] += 1 # after each episode is done, one is added to the corresponding element in decision time,
			# so after 100 episodes, we have a histogram of decision times

			if abs(next_state[1]) == args.height+1 or abs(next_state[1]) == 0: # if we made no decision till the end
				last_choice += 1 # last choice represents the number of episodes in which we waited until the end

			traj = env.get_trajectory()
			choice_made.append(_sign(next_state[1])) # these arays are updated after each episode, not after each timestep
			correct_choice.append(_sign(traj[-1]))
			finalDecisionTime.append(abs(next_state[1])) # Why next_state? because it is the latest state that we have and we don't update state until after the if-else condition
			if args.env == 'tokens-v3' or args.env == 'tokens-v4':
				finalRewardPerGame.append(env.reward)
			else:
				finalRewardPerGame.append(reward)

			traj_group.append(traj) # the list of all trajectories over all episodes
			next_state, game_time_step = env.reset()
			took_action = False
		else:
			num_frames+=1 
			update+= 1

		state = next_state


		if num_games > num_games_prevs and num_games % args.log_interval == 0: # if the game has not stpped and we moved an episode forward
			duration = int(time.time() - start_time)
			totalLoss_val = np.sum(lossPerEpisode) # sum of all episodic losses
			totalReturn_val = np.sum(totalReturns) # sum of all episodic returns

			avg_loss = np.mean(lossPerEpisode[-1000:])
			avg_returns = np.mean(totalReturns[-1000:])
			recent_correct = np.mean(numRecentCorrectChoice[-1000:])

			header = ["update", "frames", "Games", "duration"]
			data = [update, num_frames, num_games, duration] # update and num_frames are +=15 ed

			if args.softmax:
				header += ["tmp", "lr", "last"]
				data += [policy.temperature, lr, last_choice]
			else:
				header += ["eps", "lr", "last"]
				data += [policy.epsilon, lr, last_choice]

			header += ["Loss", "Returns", "Avg Loss", "Avg Returns", "Correct Percentage", "Recent Correct", "decision_time"]
			data += [totalLoss_val.item(), totalReturn_val.item(), avg_loss.item(), avg_returns.item(), numCorrectChoice/num_games, recent_correct, finalDecisionTime[num_games_prevs]]

			if args.softmax:
				txt_logger.info(
					"U {} | F {} | G {} | D {} | TMP {:.5f} | LR {:.5f} | Last {} | L {:.3f} | R {:.3f} | Avg L {:.3f} | Avg R {:.3f} | Avg C {:.3f} | Rec C {:.3f} | DT {}"
					.format(*data))
			else:
				txt_logger.info(
					"U {} | F {} | G {} | D {} | EPS {:.5f} | LR {:.5f} | Last {} | L {:.3f} | R {:.3f} | Avg L {:.3f} | Avg R {:.3f} | Avg C {:.3f} | Rec C {:.3f} | DT {}"
					.format(*data))

			# header += ["Loss", "Returns", "Avg Loss", "Avg Returns"]
			# data += [totalLoss_val, totalReturn_val, avg_loss, avg_returns]

			csv_header = ["trajectory", "choice_made", "correct_choice", "decision_time", "reward_received"]
			csv_data = [traj_group[num_games_prevs], choice_made[num_games_prevs], correct_choice[num_games_prevs], finalDecisionTime[num_games_prevs], finalRewardPerGame[num_games_prevs]]

			if num_games == 1:
				csv_logger.writerow(csv_header)
			csv_logger.writerow(csv_data)
			csv_file.flush()

			num_games_prevs = num_games

		# Save status
		if args.save_interval > 0 and num_games % args.save_interval == 0:
			# status = {"num_frames": num_frames, "update": update, "games": num_games, "totalReturns" : totalReturns}
			model.save_q_state(model_dir, num_games)
			np.save(model_dir+'/decisionTime_'+str(num_games)+'.npy', decisionTime)
			# txt_logger.info("Status saved")
			# utils.save_status(status, model_dir)

if __name__ == '__main__':
	main()