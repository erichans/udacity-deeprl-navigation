# udacity-deeprl-navigation
Navigation problem using Deep Reinforcement Learning Agent

# Project Details

![](/unity-wide.png)
## Unity ML-Agents
**Unity Machine Learning Agents (ML-Agents)** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

For game developers, these trained agents can be used for multiple purposes, including controlling [NPC](https://en.wikipedia.org/wiki/Non-player_character) behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

In this course, you will use Unity's rich environments to design, train, and evaluate your own deep reinforcement learning algorithms. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).

## The Environment
For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

<p align="center">
  <img src="/banana.gif" />
</p>

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# Getting Started

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```

2. Install pytorch >= 0.4.0

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/erichans/udacity-deeprl-navigation.git
cd udacity-deeprl-navigation/python
pip install .
```
# Instructions

## Train the Agent
```bash
python train.py
```

You can tune the model by changing the following hyperparameters in following files (default values below):

### train.py
* TOTAL_EPISODES = 500
* EPSILON_START = 1.0
* EPSILON_DECAY = .99
* EPSILON_END = 0.01
* DOUBLE_DQN = True
  * See paper [here](https://arxiv.org/abs/1509.06461)
* DUELING_DQN = True
  * See paper [here](https://arxiv.org/abs/1511.06581)
### agent.py
* LR = 5e-4 (learning rate)
* BUFFER_SIZE = 100.000
* BATCH_SIZE = 64
* GAMMA = .99 (discount factor)

* UPDATE_EVERY = 4 (How many steps to wait before update the target QNetwork)
* TAU = 1e-3 (soft update from local QNetwork parameters to target QNetwork parameters)
