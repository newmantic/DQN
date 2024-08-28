# DQN


Deep Q-Networks (DQN) is a reinforcement learning algorithm that combines Q-Learning with deep neural networks. It is designed to handle environments with large or continuous state spaces, where using a Q-table to store Q-values for each state-action pair is impractical.


Q-Learning: Q-Learning is a value-based reinforcement learning algorithm that seeks to learn the optimal action-value function, denoted as Q*(s, a). The Q-value represents the expected cumulative reward of taking action a in state s and following the optimal policy thereafter.

Q-Value Update Rule: The Q-value for a state-action pair is updated using the Bellman equation:
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max(Q(s', a')) - Q(s, a)]
where:
s is the current state.
a is the action taken.
r is the reward received after taking action a.
s' is the next state after taking action a.
alpha is the learning rate (0 < alpha <= 1).
gamma is the discount factor (0 <= gamma <= 1).
max(Q(s', a')) is the maximum Q-value over all possible actions in the next state s'.

Deep Q-Network (DQN): Instead of using a Q-table, DQN uses a deep neural network to approximate the Q-value function. The network takes the state s as input and outputs Q-values for all possible actions in that state.

Experience Replay: DQN uses a technique called experience replay to stabilize training. The agent's experiences (s, a, r, s', done) are stored in a replay memory. During training, random mini-batches of experiences are sampled from this memory to update the Q-network. This breaks the correlation between consecutive experiences and reduces the risk of divergence.

Target Network: To further stabilize training, DQN uses a target network, which is a copy of the Q-network. The target network is used to compute the target Q-values for the update:
target = r + gamma * max(Q_target(s', a'))
The weights of the target network are periodically updated to match the weights of the Q-network.


Initialize the Q-Network: Initialize the Q-network with random weights. Also, initialize the target network with the same weights.
Initialize Replay Memory: Initialize an empty replay memory to store experiences (s, a, r, s', done).

For each episode:
Initialize the starting state s.

For each step in the episode:

Choose an action a using an epsilon-greedy policy:
With probability epsilon, choose a random action (exploration).
With probability 1 - epsilon, choose the action with the highest Q-value (exploitation).
Execute the action a, observe the reward r and the next state s'.
Store the experience (s, a, r, s', done) in the replay memory.
Sample a mini-batch of experiences from the replay memory.

For each experience in the mini-batch:
If done is True (i.e., the episode ended), set the target Q-value to r.
Otherwise, set the target Q-value to:
target = r + gamma * max(Q_target(s', a'))

Update the Q-network by minimizing the loss between the predicted Q-value and the target Q-value:
Loss = (Q(s, a) - target)^2
Periodically update the target network with the weights of the Q-network.
Decay Epsilon: Reduce the exploration rate epsilon over time to shift from exploration to exploitation as the agent learns.

End of Episode:
If the episode ends (i.e., the agent reaches a terminal state), start a new episode.


Q-Network: A deep neural network approximates the Q-value function Q(s, a).
Experience Replay: Experiences are stored in a replay memory, and random mini-batches are sampled to update the Q-network. This helps stabilize training by breaking the correlation between consecutive experiences.
Target Network: A separate target network is used to compute the target Q-values, further stabilizing the learning process.
Epsilon-Greedy Policy: The agent balances exploration and exploitation using an epsilon-greedy policy, where the probability of exploring decreases over time as the agent learns.


Pros
Scalability: Can handle environments with large or continuous state spaces using deep learning.
Stability: Experience replay and target networks help stabilize training and prevent divergence.
Versatility: Applicable to a wide range of tasks, including playing video games, robotics, and more.

Cons
Computationally Intensive: Training a DQN requires significant computational resources, especially for complex environments.
Hyperparameter Sensitivity: Performance can be sensitive to the choice of hyperparameters such as learning rate, discount factor, epsilon decay, and network architecture.

