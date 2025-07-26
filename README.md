# Connect Four

In this repository I store my kaggle notebooks used to train an AI to play connect 4 with reinforcment learning. For the training I am using Stable Baselines 3. In particular, I use its PPO implementation to create and learn a model.

For evaluation there is another notebook where models play against each other and against a connect four solver that is capable to play perfectly. Additionally I implemented a Alpha Zero inspired monte carlo tree search (MCTS) that improves the playing strength of the a model.

To improve the playing strength further I could fully implement a Alpha Zero style reinforcement learning procedure. Instead I cheat a little and use games states generated from the solver as targets and train using supervised learning.
