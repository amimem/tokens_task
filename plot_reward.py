import numpy as np
import argparse
import os
import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns

pl.rc("figure", facecolor="white",figsize = (8,8))
#pl.rc("figure", facecolor="gray",figsize = (8,8))

pl.rc('text', usetex=True)
pl.rc('text.latex', preamble=[r'\usepackage{amsmath}'])
pl.rc('lines',markeredgewidth = 2)
pl.rc('font',size = 24)



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir",  required=True , help="name of the storage directory")

	args = parser.parse_args()

	num_seeds_trials = 0
	data = []

	for name in os.listdir(args.dir):
		if os.path.isdir(os.path.join(args.dir, name)):
			for root, dirs, files in os.walk(args.dir+'/'+name):
				for file in files:
					if file == 'status.pt':
						status = utils.get_status(root)
						data.append(status)

	# print(len(data))

	timesteps = len(data[0]['totalReturns'])
	colNames = ['T1', 'T2', 'T3']
	# print(timesteps)

	df_rewards = pd.DataFrame(index=range(0,timesteps), columns=colNames)
	# df_rewards['games'] = range(timesteps)

	for idx, file in enumerate(data):
		temp = np.asarray(file['totalReturns'])
		mean_arr = []
		for step, x in enumerate(temp):
			cum_mean = np.mean(temp[step-100:step])
			mean_arr.append(cum_mean)
		# cumsum = np.cumsum(temp)
		# print(len(cumsum))
		# print(temp[-100:])
		# print(len(mean_arr))
		print(idx)
		print(colNames[idx])
		df_rewards[colNames[idx]] = mean_arr

	df_rewards = df_rewards.fillna(0)
	print(df_rewards)
	print(df_rewards.std(axis=1))

				
	# ax = sns.tsplot(data = df_rewards)
	ax = sns.lineplot(data=df_rewards[1000:], dashes=False)
	# ax = sns.palplot(sns.color_palette(n_colors=1))

	# mean = v.mean(axis=1)
	# std  = v.std(axis=1)
	# ax.errorbar(df_rewards.index, mean, yerr=std, fmt='-o')
	pl.show()



	# print(len(returns))


if __name__== '__main__':
	main()