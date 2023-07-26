import gym

env= gym.make('LunarLanderContinuous-v2')
print(env.observation_space)
print(env.action_space)

#2D en action, 8D pour les états

observation=env.reset()

while(1):
    for _ in range(2000):
        env.render()
        action = env.action_space.sample()
        print(action)
        print(observation)
        observation, reward, done, info= env.step(action)
        
        if done:
            observation=env.reset()
            
env.close()

