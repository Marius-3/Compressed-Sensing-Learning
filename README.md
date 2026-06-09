# Compressed Sensing Learning

这是一个用于整理 **压缩感知（Compressed Sensing, CS）** 相关基础理论、经典算法和个人学习笔记的仓库。

---

## 目录

### 1. 贪婪类稀疏恢复算法

- [Matching Pursuit 与 Orthogonal Matching Pursuit 原理整理（PDF）](./MP_OMP.pdf)

---

## 后续计划

后续可以继续整理以下内容：

- 压缩感知基本模型；
- 稀疏性与稀疏表示；
- 字典矩阵与观测矩阵；
- Basis Pursuit，BP；
- LASSO；
- Iterative Hard Thresholding，IHT；
- CoSaMP；
- Subspace Pursuit，SP；
- Sparse Bayesian Learning，SBL；
- Bayesian Compressive Sensing，BCS；
- 雷达稀疏重构中的压缩感知模型。

---

## 基本模型

压缩感知常见观测模型可以写成 `y = Ax + n`。

其中：

- `y` 表示观测向量；
- `A` 表示观测矩阵或字典矩阵；
- `x` 表示待恢复的稀疏信号；
- `n` 表示噪声项。

压缩感知的核心思想是：如果 `x` 是稀疏的，或者在某个变换域中是稀疏的，那么可以用较少的观测数据恢复 `x`。
