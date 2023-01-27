# VFDS: Variational Foresight Dynamic Selection

Randy Ardywibowo, Shahin Boluki, Zhangyang Wang, Bobak Mortazavi, Shuai Huang, Xiaoning Qian

## Overview

In many machine learning tasks, input features with varying degrees of predictive capability are acquired at varying costs. In order to optimize the performance-cost trade-off, one would select features to observe a priori. However, given the changing context with previous observations, the subset of predictive features to select may change dynamically. Therefore, we face the challenging new problem of foresight dynamic selection (FDS): finding a dynamic and light-weight policy to decide which features to observe next, before actually observing them, for overall performance-cost trade-offs. To tackle FDS, this paper proposes a Bayesian learning framework of Variational Foresight Dynamic Selection (VFDS). VFDS learns a policy that selects the next feature subset to observe, by optimizing a variational Bayesian objective that characterizes the trade-off between model performance and feature cost. At its core is an implicit variational distribution on binary gates that are dependent on previous observations, which will select the next subset of features to observe. We apply VFDS on the Human Activity Recognition (HAR) task where the performance-cost trade-off is critical in its practice. Extensive results demonstrate that VFDS selects different features under changing contexts, notably saving sensory costs while maintaining or improving the HAR accuracy. Moreover, the features that VFDS dynamically select are shown to be interpretable and associated with the different activity types. We will release the code.

## Citation

Please consider citing our paper if you find the software useful for your work.

```
@inproceedings{ardywibowo2022vfds,
  title={VFDS: Variational Foresight Dynamic Selection in Bayesian Neural Networks for Efficient Human Activity Recognition},
  author={Ardywibowo, Randy and Boluki, Shahin and Wang, Zhangyang and Mortazavi, Bobak J and Huang, Shuai and Qian, Xiaoning},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1359--1379},
  year={2022},
  organization={PMLR}
}
```
