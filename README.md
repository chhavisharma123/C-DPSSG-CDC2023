# Code for CDC 2023 paper: "Switch and Conquer: Efficient Algorithms By Switching Stochastic Gradient Oracles For Decentralized Saddle Point Problems", Chhavi Sharma, Vishnu Narayanan, P. Balamurugan
This repository contains the codes used in CDC 2023 paper: "Switch and Conquer: Efficient Algorithms By Switching Stochastic Gradient Oracles For Decentralized Saddle Point Problems"
### Full Paper
[Switch and Conquer: Efficient Algorithms By Switching Stochastic Gradient Oracles For Decentralized Saddle Point Problems](https://arxiv.org/pdf/2309.00997.pdf)
### Abstract
We consider a class of non-smooth strongly convex strongly concave saddle point problems in a decentralized setting
without a central server. To solve a consensus formulation of
problems in this class, we develop an inexact primal dual hybrid
gradient (inexact PDHG) procedure that allows generic gradient
computation oracles to update the primal and dual variables.
We first investigate the performance of inexact PDHG with
stochastic variance reduction gradient (SVRG) oracle. Our
numerical study uncovers a significant phenomenon of initial
conservative progress of iterates of IPDHG with SVRG oracle.
To tackle this, we develop a simple and effective switching idea,
where a generalized stochastic gradient (GSG) computation
oracle is employed to hasten the iterates’ progress to a saddle
point solution during the initial phase of updates, followed by
a switch to the SVRG oracle at an appropriate juncture. The
proposed algorithm is named Decentralized Proximal Switching
Stochastic Gradient method with Compression (C-DPSSG), and
is proven to converge to an ϵ-accurate saddle point solution with
linear rate. Apart from delivering highly accurate solutions,
our study reveals that utilizing the best convergence phases
of GSG and SVRG oracles makes C-DPSSG well suited for
obtaining solutions of low/medium accuracy faster, useful for
certain applications. Numerical experiments on two benchmark
machine learning applications show C-DPSSG’s competitive
performance which validates our theoretical findings.
### Citation
```
@inproceedings{sharma2023switch,
  title={Switch and conquer: Efficient algorithms by switching stochastic gradient oracles for decentralized saddle point problems},
  author={Sharma, Chhavi and Narayanan, Vishnu and Balamurugan, P},
  booktitle={2023 62nd IEEE Conference on Decision and Control (CDC)},
  pages={1312--1319},
  year={2023},
  organization={IEEE}
}
```
### Requirements to run the code:  
-> Python version 3.7

-> Packages: numpy, networkx, math, random, matplotlib       

-> Path or file of binary classification data

-> Saddle point solution files if interested in plotting the gap between the algorithm iterates and saddle point solution
### File Descriptions
-> auc_maximization.ipynb inside AUC_maximization implements C-DPSSG algorithm for AUC maximization problem with a4a data set.

-> logistic_regression.ipynb inside Logistic_regression implements C-DPSSG algorithm for Robust Logistic Regression with a4a data set.

### For any query, reach out at
chhavisharma760@gmail.com
