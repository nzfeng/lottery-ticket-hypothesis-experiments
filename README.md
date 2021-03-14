# lottery-ticket-hypothesis-experiments

**Group members**: Eric Liang, Nicole Feng, Zhendong Yuan, Ramkumar Natarajan

**Original paper**: [*Stabilizing the Lottery Ticket Hypothesis*](https://arxiv.org/pdf/1903.01611.pdf) (2020) by Frankle, Dziugaite, Roy, Carbin

Experiments conducted for paper dissection 1 in CMU 15-780 SP21. The codebase included in the original paper was outdated and had many problems. This repo contains framework for training a simple neural net (LeNet 300-100) using iterative magnitude pruning (IMP) with rewinding. 

Training with IMP parameters `k=0` and `prune_levels=0,0,0` is equivalent to ordinary training without IMP.