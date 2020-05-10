## 1. 项目描述
利用强化学习训练蛇形机器人自主运动，分成以下几个阶段：
- 实现蛇形机器人正常步态的运动
- 实现蛇形机器人对固定目标的定位
- 实现蛇形机器人对随机目标的定位
- 实现蛇形机器人两种步态：slithering 和 sidewinding 的自动切换

## 2. 依赖环境
仿真环境：VREP，现在改名为：[CoppeliaSim](https://www.coppeliarobotics.com/) 。
依赖的包：
- PyRep: 仿真环境的 python 控制框架，github 地址：[https://github.com/stepjam/PyRep.git](https://github.com/stepjam/PyRep.git)
- baselines: openai 的一系列经典强化学习方法，github 地址：[https://github.com/openai/baselines.git](https://github.com/openai/baselines.git)
- gym：openai 的强化学习仿真环境包装框架，可以适配 baselines 中的强化学习算法。github 地址：[https://github.com/openai/gym.git](https://github.com/openai/gym.git)
- tensorflow 1.14
    - CPU 版本：```pip install tensorflow==1.14```
    - GPU 版本：```pip install tensorflow-gpu==1.14```
- numpy


