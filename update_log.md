- 每隔10个时间步长才计算奖励。
- 两种奖励：
    + 前进的步长，等于上一次距离目标的距离减去当前距离目标的距离（设为r1）
    + 当前距离目标的距离（设为r2）
- 两种奖励的耦合方式：r1 + 0.01 / power(r2, 1.8)
- 新增了 SAC 算法

**TODO:**
- 尝试不同奖励训练不同的状态函数。参考文献 [1]
- 尝试自适应添加熵正则项。（SAC 论文中提供了一种方法）



-----
**参考文献：**

[1] Van Seijen, H., Fatemi, M., Romoff, J., Laroche, R., Barnes, T., & Tsang, J. (2017). Hybrid reward architecture for reinforcement learning. Advances in Neural Information Processing Systems, 2017-Decem(Nips 2017), 5393–5403.