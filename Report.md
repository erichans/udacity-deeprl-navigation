# Learning Algorithm
**Three** algorithms were tested and **solved** the problem and they are detailed below
## [Deep Q-learning with Experience Replay](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
![](/images/deep-q-learning-algorithm.PNG)

### Model
```
Double DQN? False
Dueling? False
Local DQN -> OriginalDQN(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (fc4): Linear(in_features=32, out_features=4, bias=True)
)
Target DQN -> OriginalDQN(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (fc4): Linear(in_features=32, out_features=4, bias=True)
)
```
## [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
This algorithm changed replaced this line below:

![](/images/original-q-learning-error.PNG)

to this one to reduce Overoptimism due to estimation errors:

![](/images/double-q-learning-error.PNG)
### Model
```
Double DQN? True
Dueling? False
Local DQN -> OriginalDQN(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (fc4): Linear(in_features=32, out_features=4, bias=True)
)
Target DQN -> OriginalDQN(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (fc4): Linear(in_features=32, out_features=4, bias=True)
)
```
## [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
This algorithm changes the model from a single stream Q-network (**top**) to the dueling Q-network (**bottom**):

![](/images/dueling-networks.PNG)

The model now has two outputs: **Value** and **Advantage** and they are combined in a way to reduce the issue of identifiability between them as shown below:

![](/images/value-advantage-dueling-networks.PNG)

### Model
```
Double DQN? True
Dueling? True
Local DQN -> DuelDQN(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (fc_advantage): Linear(in_features=32, out_features=4, bias=True)
  (fc_value): Linear(in_features=32, out_features=1, bias=True)
)
Target DQN -> DuelDQN(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (fc_advantage): Linear(in_features=32, out_features=4, bias=True)
  (fc_value): Linear(in_features=32, out_features=1, bias=True)
)
```
## Common Hyperparameters used for training
* Total Episodes: 500
* Epsilon Start: 1.0
* Epsilon decay: .99
* Epsilon end: 0.01
* Learning Rate: 5e-4
* Replay Buffer size: 100.000
* Replay Buffer Sample Batch Size:  64
* Discount Factor: .99
* Target QNetwork update frequency: 4 (updates after 4 steps)
* Tau: 1e-3 (soft update from local QNetwork parameters to target QNetwork parameters)

# Plot of Rewards
## Deep Q-learning with Experience Replay
```
Episode 100/500 Average Score: 2.13, epsilon: 0.366
Episode 200/500 Average Score: 7.19, epsilon: 0.134
Episode 300/500 Average Score: 11.33, epsilon: 0.049
Episode 344/500 Average Score: 13.04, epsilon: 0.032
```
Environment solved in **244** episodes!     Average Score: **13.04**

### Detailed execution logs: [here](/results/results-original-dqn.txt)

<h3 align="center">Score evolution</h3>
<p align="center">
  <img src="/images/score-evolution-original-dqn.png" />
</p>

## Deep Reinforcement Learning with Double Q-learning
```
Episode 100/500 Average Score: 1.55, epsilon: 0.366
Episode 200/500 Average Score: 6.66, epsilon: 0.134
Episode 300/500 Average Score: 10.48, epsilon: 0.049
Episode 353/500 Average Score: 13.03, epsilon: 0.029
```
Environment solved in **253** episodes!     Average Score: **13.03**

### Detailed execution logs: [here](/results/results-ddqn.txt)

<h3 align="center">Score evolution</h3>
<p align="center">
  <img src="/images/score-evolution-ddqn.png" />
</p>


## Dueling Network Architectures for Deep Reinforcement Learning
```
Episode 100/500 Average Score: 2.15, epsilon: 0.366
Episode 200/500 Average Score: 7.89, epsilon: 0.134
Episode 300/500 Average Score: 11.71, epsilon: 0.049
Episode 345/500 Average Score: 13.06, epsilon: 0.031
```
Environment solved in **245** episodes!     Average Score: **13.06**
### Detailed execution logs: [here](/results/results-dueling-ddqn.txt)

<h3 align="center">Score evolution</h3>
<p align="center">
  <img src="/images/score-evolution-dueling-ddqn.png" />
</p>


# Ideas for Future Work

1. Hyperparameter tuning
2. Switch the Uniform Replay Buffer to [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
3. Implement the ideas from [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
4. Implement the ideas from [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
5. Implement the ideas from [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
6. Combine the implementations from 2 to 5 to implement [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
7. Switch the agent to learn from pixels
