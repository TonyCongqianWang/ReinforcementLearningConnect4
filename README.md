# Connect Four

This repository contains (kaggle) notebooks used to train and evaluate AI models playing connect 4.

For the training I first used Stable Baselines 3 (PPO in particular). Later I planned to use AlphaZero style RL but decided to directly use supervised learning with game positions acquired through self play games using a solver. 


For evaluation there is kne notebook where models play against each other and against a connect four solver that is capable to play perfectly. Additionally I implemented an Alpha Zero inspired monte carlo tree search (MCTS) that improves the playing strength of the a raw model. With this, my best models achieve almost perfect play.
