# Matching Pursuit 与 Orthogonal Matching Pursuit 原理整理

本文整理 **Matching Pursuit（MP，匹配跟踪）** 和 **Orthogonal Matching Pursuit（OMP，正交匹配跟踪）** 的基本原理。二者都是典型的贪婪稀疏恢复算法，常用于压缩感知、稀疏表示、阵列信号处理和雷达稀疏重构等问题中。

为避免 GitHub Markdown 误解析，本文尽量不用表格，并将所有重要公式单独成行。

---

## 1. 基本问题模型

MP 和 OMP 通常用于求解如下稀疏表示问题：

$$
\mathbf y = \mathbf A\mathbf x + \mathbf n
$$

其中：

- $\mathbf y \in \mathbb C^M$ 表示观测向量，也就是已知测量数据。
- $\mathbf A \in \mathbb C^{M\times N}$ 表示字典矩阵，也可以称为感知矩阵或观测矩阵。
- $\mathbf a_k \in \mathbb C^M$ 表示字典矩阵 $\mathbf A$ 的第 $k$ 列，也称为第 $k$ 个字典原子。
- $\mathbf x \in \mathbb C^N$ 表示待恢复的稀疏系数向量。
- $\mathbf n \in \mathbb C^M$ 表示噪声项。
- $M$ 表示观测维度。
- $N$ 表示字典原子个数，通常有 $N>M$。
- $K$ 表示稀疏度，即 $\mathbf x$ 中非零元素的个数。

字典矩阵可以写成：

$$
\mathbf A =
\left[
\mathbf a_1,\mathbf a_2,\ldots,\mathbf a_N
\right]
$$

其中，每一列 $\mathbf a_k$ 表示一个候选基函数、候选信号成分或候选参数单元。

稀疏恢复的目标是：从观测向量 $\mathbf y$ 中恢复稀疏系数向量 $\mathbf x$，也就是判断哪些字典原子真实存在，并估计它们对应的系数。

通常需要先将字典列归一化：

$$
\lVert \mathbf a_k \rVert_2 = 1,\quad k=1,2,\ldots,N
$$

这样做的目的是避免某些字典列由于能量较大而在相关匹配时天然占优。

---

## 2. MP 与 OMP 的共同思想

MP 和 OMP 都属于贪婪算法。

所谓贪婪，是指算法每一步都根据当前残差选择一个最匹配的字典原子，而不是一次性全局搜索所有可能的支撑组合。

两者的共同框架可以概括为：

$$
\text{当前残差}
\rightarrow
\text{选择最相关原子}
\rightarrow
\text{估计系数}
\rightarrow
\text{更新残差}
\rightarrow
\text{继续迭代}
$$

其中，残差记为：

$$
\mathbf r^{(t)}
$$

$\mathbf r^{(t)}$ 表示第 $t$ 次迭代后，观测信号 $\mathbf y$ 中尚未被已选原子解释的部分。

初始时尚未选择任何原子，因此：

$$
\mathbf r^{(0)} = \mathbf y
$$

---

## 3. Matching Pursuit，MP

### 3.1 MP 的核心思想

MP 的核心思想是：

> 每一步从字典中选择一个与当前残差最相关的原子，然后只用该原子解释当前残差的一部分，并将这一部分从残差中减去。

也就是说，MP 每次只做一次一维投影。

---

### 3.2 MP 的原子选择

第 $t$ 次迭代时，当前残差为：

$$
\mathbf r^{(t-1)}
$$

计算所有字典原子与当前残差的相关性：

$$
\mathbf a_k^H\mathbf r^{(t-1)},\quad k=1,2,\ldots,N
$$

这里：

- $(\cdot)^H$ 表示共轭转置。
- $\mathbf a_k^H\mathbf r^{(t-1)}$ 表示第 $k$ 个原子与当前残差的复内积。
- $\vert \mathbf a_k^H\mathbf r^{(t-1)} \vert$ 表示相关性幅度。

MP 选择相关性幅度最大的原子：

$$
k_t =
\arg\max_k
\vert
\mathbf a_k^H\mathbf r^{(t-1)}
\vert
$$

其中：

- $k_t$ 表示第 $t$ 次迭代选中的原子编号。
- $\arg\max$ 表示使目标函数取得最大值的索引。

因此，$\mathbf a_{k_t}$ 就是第 $t$ 次迭代中与当前残差最匹配的字典原子。

---

### 3.3 MP 的系数估计

选中原子 $\mathbf a_{k_t}$ 后，需要估计它对当前残差的贡献。

如果字典列已经归一化，即：

$$
\lVert \mathbf a_{k_t} \rVert_2 = 1
$$

则系数为：

$$
c_t =
\mathbf a_{k_t}^H\mathbf r^{(t-1)}
$$

其中：

- $c_t$ 表示第 $t$ 次迭代中选中原子的系数。
- $\mathbf a_{k_t}$ 表示第 $t$ 次迭代选中的原子。
- $\mathbf r^{(t-1)}$ 表示选择该原子之前的残差。

如果字典列没有归一化，则应写成：

$$
c_t =
\frac{
\mathbf a_{k_t}^H\mathbf r^{(t-1)}
}{
\mathbf a_{k_t}^H\mathbf a_{k_t}
}
$$

这个式子本质上表示：将当前残差投影到选中原子 $\mathbf a_{k_t}$ 所在的一维方向上。

---

### 3.4 MP 的残差更新

得到系数 $c_t$ 后，从当前残差中减去该原子的贡献：

$$
\mathbf r^{(t)}
=
\mathbf r^{(t-1)}
-
c_t\mathbf a_{k_t}
$$

其中，$c_t\mathbf a_{k_t}$ 表示第 $t$ 次迭代中被当前选中原子解释掉的信号成分。

---

### 3.5 MP 的系数向量更新

最终希望得到的是稀疏系数估计：

$$
\hat{\mathbf x}
$$

初始时：

$$
\hat{\mathbf x} = \mathbf 0
$$

如果第 $t$ 次迭代选择了第 $k_t$ 个原子，则更新：

$$
\hat{x}_{k_t}
\leftarrow
\hat{x}_{k_t}
+
c_t
$$

注意，在 MP 中，同一个原子可能被多次选中，因此这里使用累加更新。

---

### 3.6 MP 的完整流程

MP 可总结如下。

初始化：

$$
\mathbf r^{(0)}
=
\mathbf y,
\qquad
\hat{\mathbf x}
=
\mathbf 0
$$

第 $t$ 次迭代选择最相关原子：

$$
k_t =
\arg\max_k
\vert
\mathbf a_k^H\mathbf r^{(t-1)}
\vert
$$

估计当前原子系数：

$$
c_t =
\mathbf a_{k_t}^H\mathbf r^{(t-1)}
$$

其中假设 $\lVert \mathbf a_{k_t} \rVert_2=1$。

更新系数：

$$
\hat{x}_{k_t}
\leftarrow
\hat{x}_{k_t}
+
c_t
$$

更新残差：

$$
\mathbf r^{(t)}
=
\mathbf r^{(t-1)}
-
c_t\mathbf a_{k_t}
$$

重复迭代，直到满足停止条件。常见停止条件包括达到最大迭代次数：

$$
t = K
$$

或者残差足够小：

$$
\lVert \mathbf r^{(t)} \rVert_2
\leq
\varepsilon
$$

其中，$\varepsilon$ 是预设的残差阈值。

---

### 3.7 MP 的特点

MP 的特点是：

> 每一步只处理当前新选中的原子，不会重新调整之前已经选过的原子的系数。

因此 MP 的计算相对简单，但当字典原子之间存在相关性时，前面步骤产生的误差可能会在后续迭代中累积。

---

## 4. Orthogonal Matching Pursuit，OMP

### 4.1 OMP 的核心思想

OMP 是 MP 的改进版本。

OMP 的原子选择方式与 MP 相同，仍然选择与当前残差最相关的字典原子。

但 OMP 与 MP 的关键区别在于：

> OMP 每次选中新原子后，会把所有已经选中的原子放在一起，重新做一次最小二乘估计。

也就是说，OMP 不只估计当前新选原子的系数，而是对当前已选的全部原子进行联合系数估计。

---

## 5. OMP 中的支撑集

OMP 需要维护一个支撑集，记为：

$$
\mathcal S^{(t)}
$$

$\mathcal S^{(t)}$ 表示第 $t$ 次迭代后已经选中的原子编号集合。

初始时没有选择任何原子：

$$
\mathcal S^{(0)}
=
\varnothing
$$

如果第 $t$ 次迭代选中了原子编号 $k_t$，则支撑集更新为：

$$
\mathcal S^{(t)}
=
\mathcal S^{(t-1)}
\cup
\{k_t\}
$$

其中，$\cup$ 表示集合并集。

例如：

$$
\mathcal S^{(1)}
=
\{5\}
$$

表示第一次选中了第 5 个原子。

$$
\mathcal S^{(2)}
=
\{5,12\}
$$

表示前两次分别选中了第 5 个和第 12 个原子。

---

## 6. OMP 的原子选择

第 $t$ 次迭代时，OMP 同样选择与当前残差最相关的原子：

$$
k_t =
\arg\max_k
\vert
\mathbf a_k^H\mathbf r^{(t-1)}
\vert
$$

该步骤与 MP 完全相同。

区别在于，OMP 在选择完新原子后，不是只估计当前原子的系数，而是对所有已选原子进行联合最小二乘估计。

---

## 7. OMP 的最小二乘重估计

根据当前支撑集 $\mathcal S^{(t)}$，从字典矩阵 $\mathbf A$ 中取出对应列，构成子字典：

$$
\mathbf A_{\mathcal S^{(t)}}
=
[
\mathbf a_k
]_{k\in\mathcal S^{(t)}}
$$

例如，如果：

$$
\mathcal S^{(t)}
=
\{5,12,20\}
$$

则：

$$
\mathbf A_{\mathcal S^{(t)}}
=
[
\mathbf a_5,
\mathbf a_{12},
\mathbf a_{20}
]
$$

OMP 在当前支撑集上求解最小二乘问题：

$$
\hat{\mathbf x}_{\mathcal S^{(t)}}
=
\arg\min_{\mathbf z}
\lVert
\mathbf y
-
\mathbf A_{\mathcal S^{(t)}}\mathbf z
\rVert_2^2
$$

其中：

- $\mathbf A_{\mathcal S^{(t)}}$ 表示当前已选原子组成的子字典。
- $\mathbf z$ 是临时优化变量，表示当前支撑上的待求系数。
- $\hat{\mathbf x}_{\mathcal S^{(t)}}$ 表示当前支撑集上的最小二乘系数估计。
- $\lVert \cdot \rVert_2$ 表示二范数。
- $\arg\min$ 表示使目标函数取得最小值的变量。

如果 $\mathbf A_{\mathcal S^{(t)}}$ 满列秩，则最小二乘解为：

$$
\hat{\mathbf x}_{\mathcal S^{(t)}}
=
\left(
\mathbf A_{\mathcal S^{(t)}}^H
\mathbf A_{\mathcal S^{(t)}}
\right)^{-1}
\mathbf A_{\mathcal S^{(t)}}^H
\mathbf y
$$

其中：

- $(\cdot)^{-1}$ 表示矩阵求逆。
- $\mathbf A_{\mathcal S^{(t)}}^H\mathbf A_{\mathcal S^{(t)}}$ 表示子字典的 Gram 矩阵。
- $\mathbf A_{\mathcal S^{(t)}}^H\mathbf y$ 表示子字典与观测向量的相关向量。

在实际计算中，一般不显式求矩阵逆，而是使用 QR 分解、Cholesky 分解或线性方程求解器。

---

## 8. OMP 的残差更新

完成最小二乘估计后，OMP 用所有已选原子的联合拟合结果更新残差：

$$
\mathbf r^{(t)}
=
\mathbf y
-
\mathbf A_{\mathcal S^{(t)}}
\hat{\mathbf x}_{\mathcal S^{(t)}}
$$

这与 MP 明显不同。

MP 的残差更新为：

$$
\mathbf r^{(t)}
=
\mathbf r^{(t-1)}
-
c_t\mathbf a_{k_t}
$$

即只减去当前新选原子的贡献。

OMP 的残差更新为：

$$
\mathbf r^{(t)}
=
\mathbf y
-
\text{所有已选原子的联合拟合结果}
$$

因此，OMP 每次都会重新调整当前支撑集上所有原子的系数。

---

## 9. OMP 中“正交”的含义

OMP 中的 “Orthogonal” 并不是指字典矩阵 $\mathbf A$ 的列彼此正交。

它指的是：

> 每次最小二乘更新后，当前残差与所有已选原子张成的子空间正交。

数学上表示为：

$$
\mathbf A_{\mathcal S^{(t)}}^H
\mathbf r^{(t)}
=
\mathbf 0
$$

这个式子说明，经过第 $t$ 次最小二乘拟合后，当前残差已经无法被已选原子继续解释。剩余残差只包含尚未选择的成分、噪声或模型失配。

这就是 OMP 相比 MP 更稳定、更准确的关键原因。

---

## 10. 从投影角度理解 MP 与 OMP

### 10.1 MP：一维投影

MP 每一步只把当前残差投影到一个原子方向上：

$$
c_t\mathbf a_{k_t}
$$

然后从当前残差中减去这一维投影：

$$
\mathbf r^{(t)}
=
\mathbf r^{(t-1)}
-
c_t\mathbf a_{k_t}
$$

因此 MP 可以理解为：

> 每次做一次一维投影，并从残差中扣除该投影。

---

### 10.2 OMP：子空间正交投影

OMP 在第 $t$ 次迭代后，已经选择了多个原子：

$$
\mathcal S^{(t)}
=
\{k_1,k_2,\ldots,k_t\}
$$

这些原子张成一个子空间：

$$
\operatorname{span}
\left(
\mathbf A_{\mathcal S^{(t)}}
\right)
$$

OMP 将观测向量 $\mathbf y$ 投影到该子空间上：

$$
\hat{\mathbf y}^{(t)}
=
\mathbf A_{\mathcal S^{(t)}}
\hat{\mathbf x}_{\mathcal S^{(t)}}
$$

其中，$\hat{\mathbf y}^{(t)}$ 表示第 $t$ 次迭代后由已选原子重构出的信号。

残差为：

$$
\mathbf r^{(t)}
=
\mathbf y
-
\hat{\mathbf y}^{(t)}
$$

由于这是正交投影，所以：

$$
\mathbf A_{\mathcal S^{(t)}}^H
\mathbf r^{(t)}
=
\mathbf 0
$$

即当前残差与所有已选原子正交。

---

## 11. MP 与 OMP 的核心区别

MP 和 OMP 的原子选择准则相同，都是选择与当前残差最相关的原子。

二者的核心区别如下：

- MP 只估计当前新选原子的系数 $c_t$。
- OMP 对所有已选原子进行联合最小二乘估计。
- MP 的残差更新为：

$$
\mathbf r^{(t)}
=
\mathbf r^{(t-1)}
-
c_t\mathbf a_{k_t}
$$

- OMP 的残差更新为：

$$
\mathbf r^{(t)}
=
\mathbf y
-
\mathbf A_{\mathcal S^{(t)}}
\hat{\mathbf x}_{\mathcal S^{(t)}}
$$

- MP 不会重新调整旧系数。
- OMP 每次都会重新调整旧系数。
- MP 的残差不一定与所有已选原子正交。
- OMP 的残差与所有已选原子张成的子空间正交。
- MP 计算复杂度较低。
- OMP 计算复杂度较高，但恢复精度通常更好。

---

## 12. 为什么 OMP 通常比 MP 更准确？

在实际问题中，字典原子之间通常不是严格正交的。也就是说，可能存在：

$$
\mathbf a_i^H\mathbf a_j
\neq
0,
\quad
i\neq j
$$

当两个原子不正交时，某个新原子的加入会影响之前已选原子的最优系数。

MP 不会重新调整旧系数，因此可能造成误差累积。

OMP 则会在每次加入新原子后，重新求解：

$$
\hat{\mathbf x}_{\mathcal S^{(t)}}
=
\arg\min_{\mathbf z}
\lVert
\mathbf y
-
\mathbf A_{\mathcal S^{(t)}}\mathbf z
\rVert_2^2
$$

因此 OMP 能够在当前已选支撑上获得整体最优的系数估计。

可以概括为：

$$
\text{MP：逐步选原子，局部扣除贡献}
$$

$$
\text{OMP：逐步选原子，整体重估系数}
$$

---

## 13. MP 伪代码

```text
Input:
    y: observation vector
    A: dictionary matrix
    K: sparsity level or maximum number of iterations
    epsilon: residual threshold

Initialize:
    r = y
    x_hat = 0

For t = 1,2,...,K:

    1. Compute correlations:
       p = A^H r

    2. Select atom:
       k_t = argmax_k |p_k|

    3. Estimate coefficient:
       c_t = a_{k_t}^H r

    4. Update sparse coefficient:
       x_hat[k_t] = x_hat[k_t] + c_t

    5. Update residual:
       r = r - c_t a_{k_t}

    6. Stop if ||r||_2 <= epsilon

Output:
    x_hat
```

---

## 14. OMP 伪代码

```text
Input:
    y: observation vector
    A: dictionary matrix
    K: sparsity level or maximum number of iterations
    epsilon: residual threshold

Initialize:
    r = y
    S = empty set
    x_hat = 0

For t = 1,2,...,K:

    1. Compute correlations:
       p = A^H r

    2. Select atom:
       k_t = argmax_k |p_k|

    3. Update support:
       S = S union {k_t}

    4. Construct sub-dictionary:
       A_S = columns of A indexed by S

    5. Least-squares estimation:
       x_S = argmin_z ||y - A_S z||_2^2

    6. Update residual:
       r = y - A_S x_S

    7. Stop if ||r||_2 <= epsilon

Output:
    x_hat supported on S
```

---

## 15. 总结

MP 和 OMP 都通过最大相关准则逐步选择字典原子。

二者最核心的区别是：

$$
\boxed{\text{MP 每次只减去当前选中原子的贡献。}}
$$

$$
\boxed{\text{OMP 每次都会在全部已选原子上重新做最小二乘估计。}}
$$

因此，MP 是一种逐步的一维投影方法，而 OMP 是一种逐步构造支撑集并进行子空间正交投影的方法。

从直观上看：

$$
\text{MP}
=
\text{选一个，减一个}
$$

$$
\text{OMP}
=
\text{选一个，整体重拟合一次}
$$

这就是 OMP 相比 MP 的本质改进。
