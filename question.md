- entropy 目标如果加入学习目标函数，但是让其无限制增长的话，学习出的策略会过于随机化，最终导致性能下降。但是如果不加入 entropy，后期的策略提升空间不大。这是一个探索和提升的平衡问题。是否可以考虑动态地加入 entropy 目标。

- 怎么让蛇头始终朝着目标的方向？

- 怎么让蛇的运动是朝着目标的直线方向。
这个问题实际上是让功率最小的问题。自动切换步态也是让功率最小的问题。
