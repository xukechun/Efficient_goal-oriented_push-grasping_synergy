# Efficient learning of goal-oriented push-grasping synergy in clutter
This is the official repository for the paper: Efficient learning of goal-oriented push-grasping synergy in clutter.

The paper is available at https://arxiv.org/abs/2103.05405

![introductory video](images/paper_video.gif)

We focus on the task of goal-oriented grasping, in which a robot is supposed to grasp a pre-assigned goal object in clutter and needs some pre-grasp actions such as pushes to enable stable grasps. However, sample inefficiency remains a main challenge. In this paper, a goal-conditioned hierarchical reinforcement learning formulation with high sample efficiency is proposed to learn a push-grasping policy for grasping a specific object in clutter. In our work, sample efficiency is improved by two means. First, we use a goal-conditioned mechanism by goal relabeling to enrich the replay buffer. Second, the pushing and grasping policies are respectively regarded as a generator and a discriminator and the pushing policy is trained with supervision of the grasping discriminator, thus densifying pushing rewards. To deal with the problem of distribution mismatch caused by different training settings of two policies, an alternating training stage is added to learn pushing and grasping in turn. A series of experiments carried out in simulation and real world indicate that our method can quickly learn effective pushing and grasping policies and outperforms existing methods in task completion rate and goal grasp success rate by less times of motion. Furthermore, we validate that our system can also adapt to goal-agnostic conditions with better performance. Note that our system can be transferred to the real world without any fine-tuning.

![system overview](images/system_hierarchical.png)

#### Contact

Any questions, please let me know: kcxu@zju.edu.cn

## Installation

- Ubuntu 18.04
- Python 3
  - torch==1.2.0, torchvision==0.4.0
  - numpy, scipy, opencv-python, matplotlib, skimage, tensorboardX
- V-REP / CoppeliaSim (simulation environment)
- Cuda 10.1
- GTX 2080 Ti, 12GB memory is tested

Code is coming soon!