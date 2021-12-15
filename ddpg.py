#Import important libraries
import random
import gym
import copy
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from gym import wrappers
from time import time
import imageio

#q network
class QNet(nn.Module):
	def __init__(self, input_sz, action_space):
		super(QNet, self).__init__()
		self.ln1 = nn.Linear(input_sz, 256)
		self.ln2 = nn.Linear(256 + action_space, 256)
		self.ln3 = nn.Linear(256, 1)

	def forward(self, state, action):
		x = F.relu(self.ln1(state))
		x = torch.cat((x, action), dim = 1)
		x = F.relu(self.ln2(x))
		x = self.ln3(x)
		return x


#policy network
class Policy(nn.Module):
	def __init__(self, input_sz, action_space):
		super(Policy, self).__init__()
		self.ln1 = nn.Linear(input_sz, 256)
		self.ln2 = nn.Linear(256, 256)
		self.ln3 = nn.Linear(256, action_space)

	def forward(self, state):
		x = F.relu(self.ln1(state))
		x = F.relu(self.ln2(x))
		x = torch.tanh(self.ln3(x)) / 0.50
		return x


def update(x, samples):
	state_b = torch.stack([s for (s,a,r,s_p,d) in samples]).float()
	action_b = torch.stack([torch.from_numpy(a) for (s,a,r,s_p,d) in samples])
	reward_b = torch.tensor([r for (s,a,r,s_p,d) in samples])
	next_state = torch.stack([torch.from_numpy(s_p) for (s,a,r,s_p,d) in samples]).float()
	done_b = torch.tensor([int(d) for (s,a,r,s_p,d) in samples])

	target = reward_b + gamma * (1 - done_b) * QFunctionTarget(next_state, PolicyFunctionTarg(next_state)).squeeze(1)
	qfunc = QFunction(state_b, action_b.unsqueeze(1).float()).squeeze(1)

	q_loss = torch.pow(qfunc - target.detach(), 2).mean()
	Qoptim.zero_grad()
	q_loss.backward()
	torch.nn.utils.clip_grad_norm_(QFunction.parameters(),clip)
	Qoptim.step()

	policyfunc = QFunction(state_b, PolicyFunction(state_b)).squeeze(1)
	policyloss = -1 * policyfunc.mean()
	Poptim.zero_grad()
	policyloss.backward()
	torch.nn.utils.clip_grad_norm_(PolicyFunction.parameters(),clip)
	Poptim.step()

	if x % 5 == 0:
		for params, target_params in zip(QFunction.parameters(), QFunctionTarget.parameters()):
			target_params.data = (1-tau) * target_params.data + (tau) * params.data

		for params, target_params in zip(PolicyFunction.parameters(), PolicyFunctionTarg.parameters()):
			target_params.data = (1-tau) * target_params.data + (tau) * params.data

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.015, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

#ornstein uhlenbeck noise
input_sz = 3
action_space = 1
gamma = 0.99
theta = 0.15
tau = 0.005
clip = 0.15

QFunction = QNet(input_sz, action_space)
PolicyFunction = Policy(input_sz, action_space)

QFunctionTarget = copy.deepcopy(QFunction)
PolicyFunctionTarg = copy.deepcopy(PolicyFunction)

Qoptim = optim.Adam(QFunction.parameters(), lr = 1e-3)
Poptim = optim.Adam(PolicyFunction.parameters(), lr = 1e-3)


env = gym.make('Pendulum-v0')
#env = wrappers.Monitor(env, "recording", force = True)
ou_noise = OUNoise(action_space)
replay_buffer = deque(maxlen = 10000)

rewardss = []
video = []
for eps in range(1000):
	state = env.reset()
	j = 0
	ou_noise.reset()
	writer = imageio.get_writer(f'test{eps}.mp4', fps=20)

	for i in range(1000):

		Qoptim.zero_grad()
		Poptim.zero_grad()

		if eps % 50 == 0:
			env.render()
			writer.append_data(env.render(mode = 'rgb_array'))


		state1 = torch.from_numpy(state).unsqueeze(0).float()
		action = PolicyFunction(state1)
		action += torch.from_numpy(ou_noise.sample())
		action = torch.clip(action, -2,2)
		action = action.squeeze(0).detach().numpy()
		#action = noise.select_action(PolicyFunction, state1)
		
		state, reward, done,info= env.step(action)
		
		replay_buffer.append((state1.squeeze(0), action.squeeze(0), reward, state, done))

		j += reward

		if done:
			break
		#print(len(replay_buffer))
		if len(replay_buffer) > 128:

			samples = random.sample(replay_buffer,128)
			update(eps, samples)
			
		
	rewardss.append(j)
	print(j)


plt.plot(np.arange(len(rewardss)), np.array(rewardss))
plt.show()
env.close()