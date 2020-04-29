from gym.envs.registration import register

register(
	id='tokens-v0',
	entry_point='gym_tokens.envs:TokensEnv',
)

register(
	id='tokens-v1',
	entry_point='gym_tokens.envs:TokensEnv2'
)

register(
	id='tokens-v3',
	entry_point='gym_tokens.envs:TokensEnv3'
)

register(
	id='tokens-v4',
	entry_point='gym_tokens.envs:TokensEnv4'
)