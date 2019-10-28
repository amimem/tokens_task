from gym.envs.registration import register

register(
	id='tokens-v0',
	entry_point='gym_tokens.envs:TokensEnv',
)