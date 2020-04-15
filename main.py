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
	parser.add_argument("--algo", default='sarsa', help="algorithm to use: sarsa | q-learning")
	parser.add_argument("--convg", type=float, default=0.00001, help="convergence value")
	parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
	parser.add_argument("--lr_final", type=float, default=0.0001, help="learning rate")
	parser.add_argument("--save-interval", type=int, default=20000, help="number of updates between two saves (default: 30, 0 means no saving)")
	parser.add_argument("--eps_start", type=float, default=1.0, help="initial epsilon-greedy value")
	parser.add_argument("--eps_final", type=float, default=0.01, help="final epsilon-greedy value")
	parser.add_argument("--eps_games", type=int, default=40000, help="number of frames for eps greedy to go from init value to final value (default: 75k)")
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
	parser.add_argument("--height", type=int, default=15, help="game tree height")
	parser.add_argument('--fancy_discount', help='use fancy discounting rewards',action='store_true')
	parser.add_argument('--fast_block', help='fast block discounting',action='store_true')
	parser.add_argument('--fancy_eps', help='try to do epsilon-greedy per game rather than per step',action='store_true')
	parser.add_argument("--tmp_start", type=float, default=1.0, help="initial temperature value")
	parser.add_argument("--tmp_final", type=float, default=0.01, help="final temperature value")
	parser.add_argument("--tmp_games", type=int, default=10000, help="number of frames for temperature to go from init value to final value (default: 75k)")
	parser.add_argument('--softmax', help='use softmax exploration',action='store_true')


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

	numNT = (args.height * 2) + 1 
	numHT = (args.height * 2) + 1

	if args.fast_block:
		block_discount = 0.25

	else:
		block_discount = 0.75

	env = gym.make('tokens-v0', gamma=block_discount, seed=args.seed, terminal=args.height, fancy_discount=args.fancy_discount)
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
	num_games_frames = args.height * args.games

	# model = lib.Q_Table(env.get_num_states(), env.get_num_actions(), (numNT, numHT), args.convg)
	model = lib.Q_Table(numNT*numHT*(args.height+2), num_actions, (numNT, numHT, args.height), args.convg, args.height)

	if args.softmax:
		policy = lib.SoftMaxPolicy()
		tmp_track = lib.TemperatureTracker(args.tmp_start, args.tmp_final, args.tmp_games, policy)

	if args.fancy_eps:
		#TODO
		policy = lib.EpsilonGreedyGamePolicy()
		eps_track = lib.EpsilonTracker(args.eps_start,args.eps_final, args.eps_games, policy)


	else:
		policy = lib.EpsilonGreedyPolicy()
		eps_track = lib.EpsilonTracker(args.eps_start,args.eps_final, args.eps_games*args.height, policy)

	if args.algo == 'sarsa': 
		monkeyAgent = lib.SarsaAgent(policy, model, args.height)

	elif args.algo == 'q-learning':
		monkeyAgent = lib.QlAgent(policy, model, args.height)

	lr_sched = lib.LRscheduler(args.lr, args.lr_final, args.games*args.height*10*0.8)

	num_frames = status["num_frames"]
	update = status["update"]
	num_games = status["num_games"]
	num_games_prevs = 0

	start_time = time.time()
	totalReturns = [] 
	totalLoss = []

	state, game_time_step  = env.reset()
	decisionTime = []
	lossPerEpisode = []
	train_info = []

	last_choice = 0

	decisionTime = np.zeros(shape=((args.height*2)+1))
	# info = []

	traj = []
	traj_group = []

	choice_made = []
	correct_choice = []
	finalDecisionTime = []
	finalRewardPerGame = []
	numCorrectChoice = 0
	numRecentCorrectChoice = []

	while num_frames <= num_games_frames: 

		traj.append(state[0].tolist())

		if args.fancy_eps:
			eps_track.set_eps(num_games)

		else:
			eps_track.set_eps(num_frames)

		action = monkeyAgent.get_actions(state, game_time_step)

		next_state, reward, is_done, game_time_step = env.step(action)

		# print('next_state: ', next_state)
		# print('action: ', action)
		# print('is done: ', is_done)
		# print('game_time_step: ', game_time_step)
		# print('state: ', state)

		lr = lr_sched.get_lr(num_frames)
		
		next_act = monkeyAgent.get_actions(next_state, game_time_step)
		loss = model.get_TDerror(state, action, next_state, next_act, reward, args.gamma, is_done, args.algo)
		converged = model.update_qVal(lr, state, action, loss)
		totalLoss.append(loss)

		if is_done:
			num_games+=1
			totalReturns.append(reward)

			if reward > 0:
				numCorrectChoice += 1
				numRecentCorrectChoice.append(1)
			else:
				numRecentCorrectChoice.append(0)

			lossPerEpisode.append(np.sum(totalLoss))
			totalLoss = []

			decision_step = model._augState(abs(next_state[1]))
			decisionTime[decision_step-1] += 1

			if abs(next_state[1]) == args.height+1:
				last_choice += 1

			choice_made.append(_sign(next_state[1]))
			correct_choice.append(_sign(next_state[0]))
			finalDecisionTime.append(abs(next_state[1]))
			finalRewardPerGame.append(reward)

			traj_group.append(traj)
			traj = []
			next_state, game_time_step = env.reset()

		else:
			num_frames+=1 
			update+= 1

		state = next_state


		if num_games > num_games_prevs and num_games % args.log_interval == 0:
			duration = int(time.time() - start_time)
			totalLoss_val = np.sum(lossPerEpisode)
			totalReturn_val = np.sum(totalReturns)

			avg_loss = np.mean(lossPerEpisode[-1000:])
			avg_returns = np.mean(totalReturns[-1000:])
			recent_correct = np.mean(numRecentCorrectChoice[-1000:])

			header = ["update", "frames", "Games", "duration"]
			data = [update, num_frames, num_games, duration]

			header += ["eps", "lr", "last"]
			data += [policy.epsilon, lr, last_choice]

			header += ["Loss", "Returns", "Avg Loss", "Avg Returns", "Correct Percentage", "Recent Correct", "decision_time"]
			data += [totalLoss_val.item(), totalReturn_val.item(), avg_loss.item(), avg_returns.item(), numCorrectChoice/num_games, recent_correct, finalDecisionTime[num_games_prevs]]

			txt_logger.info(
				"U {} | F {} | G {} | D {} | EPS {:.3f} | LR {:.5f} | Last {} | L {:.3f} | R {:.3f} | Avg L {:.3f} | Avg R {:.3f} | Avg C {:.3f} | Rec C {:.3f} | DT {}"
				.format(*data))

			# header += ["Loss", "Returns", "Avg Loss", "Avg Returns"]
			# data += [totalLoss_val, totalReturn_val, avg_loss, avg_returns]

			csv_header = ["trajectory", "choice_made", "correct_choice", "decision_time", "reward_received"]
			csv_data = [traj_group[num_games_prevs], choice_made[num_games_prevs], correct_choice[num_games_prevs], finalDecisionTime[num_games_prevs], finalRewardPerGame[num_games_prevs]]

			# print(traj_group[num_games_prevs])
			# print(choice_made[num_games_prevs])

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