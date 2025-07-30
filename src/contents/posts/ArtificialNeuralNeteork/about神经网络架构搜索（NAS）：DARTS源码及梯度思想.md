---
title: "神经网络架构搜索（NAS）：DARTS：Differentiable Architecture Search"
published: 2025-07-19T00:00:00.000Z
description: ""
image: "src/contents/posts/ArtificialNeuralNeteork/alis1.jpg"
tags: ["AI", "DARTS", "NAS"]
category: "AI"
draft: false
lang: zh-CN
---
# *DARTS算法*

[论文源篇](https://www.bohrium.com/paper/read?libraryId=6454032&activeReadTab=chat)

## 1.基本原理：

NAS核心思想是*将架构设计过程自动化*，通过算法来探索和优化神经网络的结构。

**基本分为两步骤:**

搜索结构（在**验证集val**上进行）

在搜索好的结构上做验证（在**训练集train**上进行）

DARTS算法的核心思想在于 ***将神经网络架构搜索任务转化为一个可微分的问题** *来提高效率*。(对第一部分introducation的理解:*

**原句:**`"An inherent cause of inefficiency for the dominant approaches, e. g.  based on RL, evolution, MCTS (Negrinho & Gordon, 2017), SMBO (Liu et al. , 2018a) or Bayesian optimization (Kandasamy et al. , 2018), is the fact that architecture search is treated as a black-box optimization  problem over a discrete domain, which leads to a large number of architecture evaluations required."`

**翻译：** `对于主流方法（例如基于强化学习、演化算法、蒙特卡洛树搜索（MCTS）、顺序模型-贝叶斯优化（SMBO）或贝叶斯优化）而言，其低效率的内在原因在于，架构搜索被视为一个在离散域上的黑盒优化问题，这导致需要评估大量的架构。)`

*它使用的是如下两个方法:*

### ***1*.连续松弛架构表示（Continuous Relaxation of Architecture Representation）** ：

神经网络的架构不再被视为一系列离散的选择（例如，选择卷积操作或池化操作）。

相反，它将每个操作的连接和操作类型都表示为 **连续的加权和** 。这意味着在搜索过程中，每个可能的连接和操作都会被赋予一个可学习的权重。

例如，在一个节点node上，如果可以有多种操作（如3x3卷积、5x5卷积、池化等），DARTS会为每个操作分配一个权重，最终的输出是所有操作输出的加权和。这些权重决定了每个操作对最终架构的贡献程度。

---

***下面用论文中的公式解释:***

* $x^{(j)}=\sum_{i<j}^{}o^{(i,j)}(x^{(i)}) $

  * $x^{(i)}$就是node(***节点***,根据原文的说法：**潜在表示（latent representation）例如，卷积网络中的特征图）**),
  * 操作 $o^{(i,j)}$代表从node(节点)  i 到node(节点) j 的操作,(i,j)可以理解为**有向边**,操作$o^{(i,j)}$是从一组备选操作 $ (如本篇论文:$3\mathfrak{\times } 3\mathfrak{conv}$ $5\mathfrak{\times } 5\mathfrak{conv}$ $3\mathfrak{\times } 3\mathfrak{max pool}$ $3\mathfrak{\times } 3\mathfrak{avg pool}$等,应该可以对应到之后的O)中进行选择,每条有向边 (**(**i**,**j**)** 都与某个操作 $o^{(i,j)}$ 相关联，该操作转换 $x^{(i)}$。
  * 然后就是原文说了一些其他的东西:
  * `For convolutional cells, the input nodes are defined as the cell outputs in the previous two layersFor recurrent cells, these are defined as the input atthe current step and the state carried from the previous step. The output of the cell is obtained byapplying a reduction operation (e.g. concatenation) to all the intermediate nodes.`
  * 我们假设单元有两个输入节点和一个输出节点。对于卷积单元，输入节点定义为前两层的单元输出 。对于循环单元，输入节点定义为当前步的输入和前一步传递的状态。单元的输出通过对所有中间节点应用归约操作（例如，连接）获得。
  * Each intermediate node is computed based on all of its predecessors:
  * 每个中间节点都基于其所有前驱节点()进行计算：
  * $x^{(j)}=\sum_{i<j}^{}o^{(i,j)}(x^{(i)})$~~对，就是上边那个公式~~
  * 那还有特殊情况，有的时候，两个节点不用有边，这个论文也有提到（其实原文的图解释更方便）~~~~*
  * *`A special zero operation is also included to indicate a lack of connection between two nodes. The task of learning the cell therefore reduces to learning the operations on its edges`*
  * 还包含一个特殊的***零操作***，表示两个节点之间没有连接。因此，学习单元的任务就简化为学习其边上的操作。
  * 关于**零操作 (Zero Operation)**
  * 为了处理节点之间没有连接的情况，在候选操作集 O 中包含了一个特殊的“零操作” (zero operation)。这个操作的输出是零，相当于断开了连接。
  * 通过学习零操作的 **α** 值，模型自动决定哪些连接应该被保留，哪些应该被剪除。如果零操作的 **α** 值最高，那么这条边实际上就没有连接。(*~~就是把零操作也归为一个选项，和上面的卷积一类的操作一样，算概率去选择,隶属于O~~*)
* $$
  O
  $$

  对于一对节点 **(**i**,**j**)** 之间的连接(就是之前说的***有向边***)，如果有一组候选操作 **O(原文对O的解释是:设 O是候选操作的集合（例如，卷积、最大池化、零），其中每个操作表示应用于 x(i)的某个函数 o(⋅)**

  对应文件:`genotypes.py`(对应操作文件:`operations.py文件`): `cnn/genotypes.py`

  ```python
  from collections import namedtuple

  Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

  PRIMITIVES = [
      'none',           # 零操作
      'max_pool_3x3',   # 3x3最大池化
      'avg_pool_3x3',   # 3x3平均池化
      'skip_connect',   # 跳跃连接
      'sep_conv_3x3',   # 3x3深度可分离卷积
      'sep_conv_5x5',   # 5x5深度可分离卷积
      'dil_conv_3x3',   # 3x3空洞卷积
      'dil_conv_5x5'    # 5x5空洞卷积
  ]

  NASNet = Genotype(
    normal = [
      ('sep_conv_5x5', 1),
      ('sep_conv_3x3', 0),
      ('sep_conv_5x5', 0),
      ('sep_conv_3x3', 0),
      ('avg_pool_3x3', 1),
      ('skip_connect', 0),
      ('avg_pool_3x3', 0),
      ('avg_pool_3x3', 0),
      ('sep_conv_3x3', 1),
      ('skip_connect', 1),
    ],
    normal_concat = [2, 3, 4, 5, 6],

    reduce = [
      ('sep_conv_5x5', 1),
      ('sep_conv_7x7', 0),
      ('max_pool_3x3', 1),
      ('sep_conv_7x7', 0),
      ('avg_pool_3x3', 1),
      ('sep_conv_5x5', 0),
      ('skip_connect', 3),
      ('avg_pool_3x3', 2),
      ('sep_conv_3x3', 2),
      ('max_pool_3x3', 1),
    ],
    reduce_concat = [4, 5, 6],
  )
  # 普通单元包含10个操作,输出拼接5个节点（2-6）,降采样单元输出拼接3个节点（4-6）
  AmoebaNet = Genotype(
    normal = [
      ('avg_pool_3x3', 0),
      ('max_pool_3x3', 1),
      ('sep_conv_3x3', 0),
      ('sep_conv_5x5', 2),
      ('sep_conv_3x3', 0),
      ('avg_pool_3x3', 3),
      ('sep_conv_3x3', 1),
      ('skip_connect', 1),
      ('skip_connect', 0),
      ('avg_pool_3x3', 1),
      ],
    normal_concat = [4, 5, 6],
    reduce = [
      ('avg_pool_3x3', 0),
      ('sep_conv_3x3', 1),
      ('max_pool_3x3', 0),
      ('sep_conv_7x7', 2),
      ('sep_conv_7x7', 0),
      ('avg_pool_3x3', 1),
      ('max_pool_3x3', 0),
      ('max_pool_3x3', 1),
      ('conv_7x1_1x7', 0),
      ('sep_conv_3x3', 5),
    ],
    reduce_concat = [3, 4, 6]
  )

  #     大量使用池化操作,普通单元输出拼接3个节点（4-6）,降采样单元输出拼接3个节点（3,4,6）
  DARTS_V1 = Genotype(normal=[  
      ('sep_conv_3x3', 1),  # 操作1：节点1→新节点，使用sep_conv_3x3
      ('sep_conv_3x3', 0),  # 操作2：节点0→新节点，使用sep_conv_3x3
      ('skip_connect', 0),  # 操作3：节点0→新节点，使用跳跃连接
      ('sep_conv_3x3', 1),  # 操作4：节点1→新节点，使用sep_conv_3x3
      ('skip_connect', 0),  # 操作5：节点0→新节点，使用跳跃连接
      ('sep_conv_3x3', 1),  # 操作6：节点1→新节点，使用sep_conv_3x3
      ('sep_conv_3x3', 0),  # 操作7：节点0→新节点，使用sep_conv_3x3
      ('skip_connect', 2)   # 操作8：节点2→新节点，使用跳跃连接], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
  DARTS_V2 = Genotype(
    normal=[
      ('sep_conv_3x3', 0),  # 操作1：节点0→新节点
      ('sep_conv_3x3', 1),  # 操作2：节点1→新节点
      ('sep_conv_3x3', 0),  # 操作3：节点0→新节点
      ('sep_conv_3x3', 1),  # 操作4：节点1→新节点
      ('sep_conv_3x3', 1),  # 操作5：节点1→新节点
      ('skip_connect', 0),  # 操作6：节点0→新节点
      ('skip_connect', 0),  # 操作7：节点0→新节点
      ('dil_conv_3x3', 2)   # 操作8：节点2→新节点
    ],
  normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

  DARTS = DARTS_V2  # 设置默认架构

  ```

  `rnn/genotypes.py`

  ```python

  from collections import namedtuple
  # 定义神经网络架构的基因表示
  Genotype = namedtuple('Genotype', 'recurrent concat')
  # recurrent：递归单元中每一步使用的操作及其前驱节点

  # concat：哪些节点的输出将被拼接（通常用于形成最终隐藏状态）
  PRIMITIVES = [
      'none',  # 不连接（丢弃）
      'tanh',  # 常用非线性激活函数
      'relu',  # 更现代的激活函数
      'sigmoid',  # 通常用于门控机制（可被选择）
      'identity'  # 恒等映射（跳跃连接）
  ]

  STEPS = 8
  CONCAT = 8
  # 构造的 cell 中有 8 个中间节点（hidden state）

  # 最终输出为这些节点的拼接（或加权）
  ENAS = Genotype(
      recurrent = [
          ('tanh', 0),
          ('tanh', 1),
          ('relu', 1),
          ('tanh', 3),
          ('tanh', 3),
          ('relu', 3),
          ('relu', 4),
          ('relu', 7),
          ('relu', 8),
          ('relu', 8),
          ('relu', 8),
      ],
      concat = [2, 5, 6, 9, 10, 11]
  )

  DARTS_V1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4), ('identity', 1), ('relu', 5), ('relu', 1)], concat=range(1, 9))
  DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))

  DARTS = DARTS_V2

  ```
* $$
  {a_{}^{(i,j)}}
  $$

  可以看作一个**集合**或**向量**,对于网络中的任意一对节点(**i**,**j**) 之间的连接，以及给定的候选操作集合 **O**，都会有一个对应于这条连接的**向量.**

  这个向量 **α**(**i**,**j**) 的维度等于候选操作的数量 **∣**O**∣**。向量中的每一个元素 ${a_{o}^{(i,j)}}$ **都代表了操作 o** 在连接 **(**i**,**j**)** 上的**强度** 。

  ~~*~~* 原文是alpha,不是a,但不必在意 ~~~~
* $$
  α={α_{}^{(i,j)}}
  $$

  **α**的整体形式可以表示为：

  **α**=**{**α**(**i**,**j**)**∣**对于网络中所有的** **(**i**,**j**)}    (我的理解)**
* $$
  \frac{e^{a_{o}^{(i,j)}}}{\sum_{o'\in O}e^{a_{o'}^{(i,j)}}}
  $$

  你可以理解为某一个操作o在i和j两节点上的归一化权重,

  **softmax 函数**

  也就是上面的公式,将原始的 α 权重转换为一个概率分布,归一化的权重。(这意味着对于同一条边，所有操作的权重之和为 1。)
* $$
  \bar{o}^{(i,j)}(x)=\sum_{o\in O}\frac{e^{a_{o}^{(i,j)}}}{\sum_{o'\in O}e^{a_{o'}^{(i,j)}}}o(x)
  $$

  * **可以理解为利用 $a= \left\{ a^{(i,j)}\right\}$(**可学习的架构参数**,是可训练的,并且由于 softmax 函数和加权和都是可微的，所以我们可以通过梯度下降来优化这些 **α** 值)做softmax**激活，然后对各操作作加权平均
  * then,这个连接上的***混合操作***$\bar{o}^{(i,j)}(x)$ 被定义为所有候选操作 **o**(**x**) 的加权和(整个架构变得“可微”,当计算损失函数对架构参数 **α** 的梯度时，可以通过链式法则，沿着这个混合操作反向传播梯度)
* 对应操作 o(有向边选择?)的**权重**(或称为“架构参数”），通过 softmax 函数将其归一化为概率分布。至此,每个连接上的操作不再是单一的，而是所有候选操作的加权组合。于是,架构本身（即所有的 **α** 值）就变成了一组连续的变量。
* ~~个人对连续松弛的理解就是：不是选或不选某个操作，而是允许在每个连接上同时存在所有候选操作的“混合”,在每一个节点上利用算法分析个权重，权重最大的就让算法选.然后softmax之所以用exp是为了连续可微,保持其可微性~~

***`After relaxation, our goal is to jointly learn the architecture α and the weights w within all the mixed ***operations (e.g. weights of the convolution ﬁlters). Analogous to architecture search using RL (Zoph & Le, 2017; Zoph et al., 2018; Pham et al., 2018b) or evolution (Liu et al., 2018b; Real et al., 2018) where the validation set performance is treated as the reward or ﬁtness, DARTS aims to optimize the validation loss, but using gradient descent`******

***松弛后，我们的目标是共同学习架构 α 和所有混合操作中的权重 w（例如卷积滤波器的权重）。与使用强化学习
 (RL) (Zoph & Le, 2017; Zoph et al., 2018b) 或演化 (Liu et al., 2018b;
Real et al., 2018) 的架构搜索类似，其中验证集性能被视为奖励或适应度，DARTS 旨在优化验证损失，但使用梯度下降。***

---

### 关于代码

#### *搜索空间定义（`model_search.py`*）:

##### ***(1)操作混合 (MixedOp)***:

将离散操作连续化，使用Softmax加权混合操作

实现公式：$\bar{o}^{(i,j)}(x)=\sum_{o\in O}\frac{e^{a_{o}^{(i,j)}}}{\sum_{o'\in O}e^{a_{o'}^{(i,j)}}}o(x)$

```python
# model_search.py
class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    #PRIMITIVES中有8个操作
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:#给池化操作后面加一个BatchNorm2d
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)#将这些操作都放在预先定义好的modulelist中



  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))
  ##op(x)就是对输入x做一个相应的操作   w1*op1(x)+w2*op2(x)+...+w8*op8(x)
  #也就是对输入x做8个操作并乘以相应的权重，把结果加起来

#i=0     4 4 48 48 16 False False
      #i=1      4 4 48 64 16 False False
      #i=2      4 4 64 64 32 True False
      #i=3     4 4 64 128 32 False True
      #i=4    4 4 128 128 32 False False
      #i=5    4 4 128 128 64 True False
      #i=6    4 4 128 256 64 False True
      #i=7    4 4 256 256 64 False False
```

##### (2)Cell（神经网络单元）

```python
# model_search.py
class Cell(nn.Module):#4 4 48 48 16 False False 每个单元中操作的步数 每个单元输出时要拼接的中间状态的数量
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction #False
    #preprocess0和preprocess1 分别代表两个前驱节点
    if reduction_prev: #前一个单元是否为降采样单元
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    self._steps = steps #4
    self._multiplier = multiplier#最终输出时要拼接的中间状态的数量

    self._ops = nn.ModuleList()

    self._bns = nn.ModuleList()

    #经历4个intermediate nodes构建混合操作
    for i in range(self._steps):
      #节点i之前的所有前驱节点
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        #构建两个节点之间的混合操作
        op = MixedOp(C, stride)
        self._ops.append(op)
        #self._ops总共包含14次MixedOp

    # logging.info(self._ops)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]#当前节点的前驱节点
    offset = 0
    #len(weights)=14
    for i in range(self._steps):
    # 遍历每个intermediate nodes，得到每个节点的output
      #i=0 根据前驱节点s0,s1计算sum和
      #b1=s0经过一次MixedOp里面的所有操作+s1经过一次MixedOp里面的所有操作
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))# torch.tensor 这一步对应两个前驱节点经过的操作+
      #s为当前节点i的output，在ops找到i对应的操作，然后对i的所有前驱节点做相应的操作（调用了MixedOp的forward），然后把结果相加
      offset += len(states)
      states.append(s)#将当前节点i的output作为下一个节点的输入
      # 选取最后四个元素作为输出
    return torch.cat(states[-self._multiplier:], dim=1)#对intermediate的output进行concat作为当前cell的输出
                                                       #dim=1是指对通道这个维度concat，所以输出的通道数变成原来的4倍

```

***每个Cell包含***：

* 2个输入节点 (s0, s1)
* 4个中间节点 (steps=4)
* 输出节点（最后4个节点的拼接）

***连接规则** *：

* 节点0：2个前驱 (s0, s1)
* 节点1：3个前驱 (s0, s1, 节点0)
* 节点2：4个前驱 (+节点1)
* 节点3：5个前驱 (+节点2)

***通道变化** *：

* 输出通道 = 4 × 当前通道 (multiplier=4)

***输出** ：*

* *最后4个节点通道拼接*

##### *(3). Network（整体架构）*

定义了可微架构搜索的整体结构

```python
class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        # Stem层
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
  
        # 构建8层Cell
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:  # 第3层和第6层降采样
                C_curr *= 2
                reduction = True
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
  
        # 分类器
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
  
        # 初始化架构参数α
        self._initialize_alphas()
  
    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))  # k=14
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, len(PRIMITIVES)))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, len(PRIMITIVES)))
  
    def genotype(self):
        def _parse(weights):
            gene = []
            start = 0
            for i in range(self._steps):  # 处理每个中间节点
                end = start + i + 2
                W = weights[start:end]  # 当前节点的所有前驱边
    
                # 选择top2边（排除none操作）
                edges = sorted(range(len(W)), 
                              key=lambda x: -max(W[x][k] for k in range(len(W[x])) 
                                           if k != PRIMITIVES.index('none')))[:2]
    
                # 选择每条边的最佳操作
                for j in edges:
                    k_best = max(range(len(W[j])), 
                                key=lambda k: W[j][k] if k != PRIMITIVES.index('none') else -float('inf'))
                    gene.append((PRIMITIVES[k_best], j))
    
                start = end
            return gene
  
        # 解析normal cell和reduction cell
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
  
        return Genotype(
            normal=gene_normal, normal_concat=range(2, 6),
            reduce=gene_reduce, reduce_concat=range(2, 6)
        )
```

 **核心组件** ：

1. **Stem层** ：初始卷积（3→48通道）
2. **Cell堆叠** ：8层Cell（含2层降采样）
3. **架构参数** ：

* `alphas_normal`：普通单元的14×8参数矩阵
* `alphas_reduce`：降采样单元的14×8参数矩阵

4. **离散化方法** ：

* 对每个中间节点选择权重最高的两条边
* 对每条边选择权重最高的操作（排除none操作）
* 输出拼接节点2-5（索引范围2-5）

***model.py***

```python
import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path

# Cell 类：固定架构单元
class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        # 预处理输入节点
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
  
        # 从基因型解析操作
        if reduction:
            op_names, indices = zip(*genotype.reduce)  # 降采样单元操作
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)  # 普通单元操作
            concat = genotype.normal_concat
  
        # 编译单元结构
        self._compile(C, op_names, indices, concat, reduction)
  
    def _compile(self, C, op_names, indices, concat, reduction):
        self._steps = len(op_names) // 2  # 每个节点有2个输入
        self._concat = concat
        self.multiplier = len(concat)  # 输出拼接节点数
  
        self._ops = nn.ModuleList()
        # 实例化所有操作
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1  # 前两个节点降采样
            op = OPS[name](C, stride, True)  # 创建操作（带affine参数的BN）
            self._ops.append(op)
  
        self._indices = indices  # 保存连接索引
  
    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)  # 预处理输入0
        s1 = self.preprocess1(s1)  # 预处理输入1
  
        states = [s0, s1]
        # 遍历每个中间节点（每节点2个输入）
        for i in range(self._steps):
            # 获取两个输入节点
            h1 = states[self._indices[2*i]]     # 第一个输入节点
            h2 = states[self._indices[2*i+1]]   # 第二个输入节点
  
            # 获取对应操作
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
  
            # 应用操作
            h1 = op1(h1)
            h2 = op2(h2)
  
            # 应用DropPath正则化
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
  
            # 节点输出 = 输入1 + 输入2
            s = h1 + h2
            states.append(s)  # 添加到状态列表
  
        # 拼接指定节点作为输出
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=3),  # 8x8 → 2x2
            nn.Conv2d(C, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, 2),     # 2x2 → 1x1
            nn.BatchNorm2d(768),
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=2),  # 14x14 → 5x5
            nn.Conv2d(C, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, 2),     # 5x5 → 4x4
            # 注意：原始实现省略了此处的BN
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x


class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super().__init__()
        # Stem层 (3 → 48通道)
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
  
        # 构建Cell堆叠
        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
  
        for i in range(layers):
            # 在第1/3和2/3层降采样
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
  
            # 创建Cell
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells.append(cell)
  
            # 更新通道数
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
  
            # 在第2/3层初始化辅助分类器
            if i == 2*layers//3:
                C_to_auxiliary = C_prev
  
        # 辅助分类器
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
  
        # 分类器
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
  
    def forward(self, x):
        # 前向传播
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            # 在第2/3层计算辅助输出
            if i == 2*self._layers//3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
  
        # 主分类器
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super().__init__()
        # 更复杂的Stem层
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C//2, 3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(C//2),
            nn.ReLU(),
            nn.Conv2d(C//2, C, 3, stride=2, padding=1),   # /4
            nn.BatchNorm2d(C)
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C, C, 3, stride=2, padding=1),     # /8
            nn.BatchNorm2d(C)
        )
  
    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev
        if auxiliary:
      		self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        # 分类器
        self.global_pooling = nn.AvgPool2d(7)  # 224x224 → 32x32 → 7x7
        self.classifier = nn.Linear(C_prev, num_classes)
  
    def forward(self, x):
        s0 = self.stem0(x)  # 224x224 → 56x56
        s1 = self.stem1(s0) # 56x56 → 28x28
        for i, cell in enumerate(self.cells):
      		s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      		if i == 2 * self._layers // 3:
        		if self._auxiliary and self.training:
          			logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
    	logits = self.classifier(out.view(out.size(0), -1))
    	return logits, logits_aux

```

---

## 2.双层优化 (Bilevel Optimization)

将架构参数 α 连续化后，使用双层优化来联合***优化架构alpha和网络权重w***。

* **网络权重 w**：神经网络中传统的可训练参数（如卷积核的权重）。
* **架构参数 α**：控制操作混合的权重。
* **训练损失$`\boldsymbol{L}_{train}\left ( \omega ,\alpha  \right )`$** ：衡量在给定架构 **α** 和权重 **w** 下模型在训练集上的性能。
* **验证损失 $`\boldsymbol{L}_{val}(\omega ^{*}(\alpha ),\alpha )`$**：衡量在给定架构 α 和权重 **w** 下模型在验证集上的性能。

由于架构表示被松弛为连续的，并且操作的权重是可学习的参数，因此整个架构搜索过程可以被视为一个 **双层优化（Bilevel Optimization）** 。

### ***1. 下层问题 (Lower-Level Problem)：权重优化(**内部优化**)***

* 目标：给定一个架构 **α(也就是上面的$α={α_{}^{(i,j)}}$)**，找到使训练损失 **L**train 最小化的网络权重 **w**。
* 表示：$\omega ^{\ast }=argmin_{\omega }\boldsymbol{L}_{train}\left ( \omega ,\alpha  \right )$
* 这表示对于任何给定的架构，我们都假设它会被训练到在训练集上表现最佳
* **w** 是下层变量

### *2.**上层问题 (Upper-Level Problem)**:**外部优化 (Upper-level optimization)***

尝试不同的架构，然后评估它们在验证集上的性能，并选择性能最好的架构。

* **目标** ：找到使**验证损失 (Lval**L**val****)** 最小化的架构参数 **α**。
* **数学表示** ：$\underset{\alpha }{min} \boldsymbol{L}_{val}(\omega ^{*}(\alpha ),\alpha )$
* **含义** ：
* 这里的 α 就是我们之前的 **可学习的架构参数**
* 上层问题的关键在于，它优化的目标是 **验证损失** 。这确保了学习到的架构具有良好的泛化能力，避免了在训练集上过拟合。
* 最重要的一点是，验证损失 Lval**L**val 的计算 **依赖于下层问题得到的 $`(\omega ^{*}(\alpha )`$**。也就是说，我们不是直接优化 **α**，而是优化在给定 **α** 下，训练好的模型在验证集上的表现。

### *3.与 **α** 和 $\bar{o}^{(i,j)}(x)$ 的关系*

#### (1).**α**

**从之前的分析我们知道α 定义了架构**

而上层问题就是关于 **α** 的优化

#### (2)$\bar{o}^{(i,j)}(x)$

**混合操作**的公式，将离散的操作选择转化为连续的加权和,这是我们知道的。

是**实现 α** **可微优化**的**具体机制**

### *4.近似架构梯度:高效运行算法:二级优化问题*

已知;

上层优化需要计算 ${∇}_{α} \boldsymbol{L}_{val}(\omega ^{*}(\alpha ),\alpha )$。这里的 $`\omega ^{*}(\alpha )`$ 是下层优化（训练 **w** 直到收敛）的解。

BUT:

每次更新 **α** 之前，都必须将网络权重 **w** 训练到在训练集上完全收敛，计算资源和时间,成本高，且难以精确求解。

因此:

#### *提出:**近似**$`\omega ^{*}(\alpha )`$*

即:

不等待w完全收敛，而是用**仅仅一步（或少数几步）梯度下降更新后的权重**来近似$`\omega ^{*}(\alpha )`$

*`(Evaluating the architecture gradient exactly can be prohibitive due to the expensive inner optimization.We therefore propose a simple approximation scheme as follows)`*

对应公式（5）（6）:

$$
{∇}_{α} \boldsymbol{L}_{val}(\omega ^{*}(\alpha ),\alpha )

≈{∇}_{α} \boldsymbol{L}_{val}(\omega-ξ{∇}_{\omega}\boldsymbol{L}_{train}(\omega,\alpha), \alpha)
$$

* ***ξ:一个小的学习率通常与 w 的学习率相关***
* ***${∇}_{\omega}\boldsymbol{L}_{train}(\omega,\alpha)$:当前架构 α 下，训练损失对 w 的梯度。***
* ***$\omega-ξ{∇}_{\omega}\boldsymbol{L}_{train}(\omega,\alpha)$:“一步优化”的"奇技淫巧",即  :  $`{∇}_{\omega}\boldsymbol{L}_{train}(\omega,\alpha)=0`$,w到达局部极值点时,w足够近似 $\omega ^{*}(\alpha )$来计算架构梯度***

#### 用***链式法则***对其进行展开(毕竟算子的本质还是微分):

$$
{∇}_{α} \boldsymbol{L}_{val}(\omega-ξ{∇}_{\omega}\boldsymbol{L}_{train}(\omega,\alpha), \alpha)
={∇}_{α} \boldsymbol{L}_{val}(\omega',α)-\xi ∇_{\alpha,\omega  }^{2} \boldsymbol{L}_{train}(w,α)∇_{\omega '}\boldsymbol{L}_{val}(\omega' ,\alpha )
$$

其中:

$\omega'=\omega-ξ{∇}_{\omega}\boldsymbol{L}_{train}(\omega,\alpha)$

* ${∇}_{α} \boldsymbol{L}_{val}(\omega',α)$:                                                                 验证损失 $\boldsymbol{L}_{val}$ 对 **α** 的直接梯度，假设 **w**′ 是固定的.可通过一次反向传播计算
* $-\xi ∇_{\alpha,\omega  }^{2} \boldsymbol{L}_{train}(w,α)∇_{\omega '}\boldsymbol{L}_{val}(\omega' ,\alpha )$:                                  这一项包含了**二阶导数** $`∇_{\alpha,\omega  }^{2} \boldsymbol{L}_{train}(w',α)`$，它表示训练损失对 *α* 和 **w** 的混合二阶导数（Hessian-vector product）
  ***直接计算二阶导数矩阵非常耗时,故论文利用多元函数泰勒展开实现了如下近似(**有限差分近似二阶导数项**):***

$$
∇_{\alpha,\omega  }^{2} \boldsymbol{L}_{train}(w',α)≈\frac{∇_{\alpha} \boldsymbol{L}_{train}(w^{+},α)-∇_{\alpha} \boldsymbol{L}_{train}(w^{-},α)}{2\epsilon }
$$

* **解释** ：
* * 这里的 **ϵ** 是一个很小的标量。
* * $\omega ^{\pm } =\omega \pm \epsilon {∇}_{w'} \boldsymbol{L}_{val}(\omega',α)$`
* * **计算过程** ：

  1. 首先计算 ${∇}_{w'} \boldsymbol{L}_{val}(\omega',α)$（验证损失对 **w**′ 的梯度）。
  2. 然后，根据这个梯度，微扰 **w** 得到 **w**+ 和 **w**−。
  3. 分别计算${∇_{\alpha} \boldsymbol{L}_{train}(w^{+},α)} $以及${∇_{\alpha} \boldsymbol{L}_{train}(w^{-},α)}$ 训练损失对 α 的梯度，在微扰后的 **w** 下）。
  4. 通过它们的差值除以 **2**ϵ 来近似二阶导数项。

#### ***一阶近似   ,  二阶近似 :***

`When ξ = 0, the second-order derivative in equation 7 will disappear. In this case, the architecture gradient is given by ∇αLval(w, α), corresponding to the simple heuristic of optimizing the validation loss by assuming the current w is the same as w∗(α). This leads to some speed-up but empirically worse performance, according to our experimental results in Table 1 and Table 2. In the following, we refer to the case of ξ = 0 as the ﬁrst-order approximation, and refer to the gradient formulation with ξ > 0 as the second-order approximation.`

* **一阶近似** ：当 ξ=0 时，近似架构梯度公式中的第二项（包含二阶导数）会消失。此时，$`∇αLval(w − ξ∇wLtrain(w, α), α)` $简化为$ `∇αLval(w, α)`$。这相当于假设当前的 w**w** 已经足够好，直接优化验证损失对 α**α** 的梯度。这种方法更快，但通常性能稍差。
* **二阶近似** ：当 ξ>0**ξ**>**0** 时，使用上述完整的包含二阶导数项的公式。虽然计算量稍大，但实验证明能够获得更好的性能
* ~~DARTS 论文中的主要结果都是基于二阶近似。~~

### *5.**工作流程***

**1.初始化** ：随机初始化架构参数 **α** 和网络权重 **w**。

*`Create a mixed operation `$\bar{o}^{(i,j)}$ `parametrized by α(i,j) for each edge (i, j)`*

**2.下层优化（权重更新）** ：

* 使用当前的 **α** 值，定义网络中的所有混合操作 $`\bar{o}^{(i,j)}(x)`$。
* 在这个由 $`\bar{o}^{(i,j)}(x)`$ **构成的网络上，使用训练数据和训练损失 $`\boldsymbol{L}_{train}$`，通过梯度下降更新网络权重 w**。在实际中，DARTS 使用了近似方法，只进行一步或几步权重更新，而不是完全收敛。

**3.上层优化（架构更新）** ：

* 使用更新后的权重 **w**，在验证数据上计算验证损失 Lval**L**val。
* 计算验证损失对架构参数 **α** 的梯度 ${∇}_{α} \boldsymbol{L}_{val}$。这个梯度是通过链式法则，反向传播经过 $`\bar{o}^{(i,j)}(x)`$ 得到的。
* 使用这个梯度来更新架构参数 **α**。

**4.重复** ：重复步骤 2 和 3，直到 **α** 收敛。

**5.导出** ：当 **α** 收敛后，通过选择每个连接上 **α** 值最高的那个操作，导出最终的离散架构。

#### 关于代码:

对应文件architect.py

##### 1. **核心函数与类:**

***`_concat(xs):`***

将张量列表展平并拼接为一个一维张量,

处理模型参数的梯度/权重向量化

```python
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])
```

***`_clip(grads, max_norm):`***

裁剪梯度,计算裁剪系数：clip_coef = max_norm / total_norm

```python
def _clip(grads, max_norm):
    total_norm = sum(g.data.norm(2)**2 for g in grads) ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return clip_coef
```

##### 2. **Architect 类：**

架构优

###### 初始化 `__init__`

```python
def __init__(self, model, args):
    self.network_weight_decay = args.wdecay  # 权重衰减系数
    self.network_clip = args.clip            # 梯度裁剪阈值
    self.model = model
    # 为架构参数α创建Adam优化器
    self.optimizer = torch.optim.Adam(
        self.model.arch_parameters(), 
        lr=args.arch_lr, 
        weight_decay=args.arch_wdecay
    )
```

###### 核心方法 `_compute_unrolled_model`

```python
def _compute_unrolled_model(self, hidden, input, target, eta):
    # 1. 计算损失和梯度
    loss, hidden_next = self.model._loss(hidden, input, target)
    theta = _concat(self.model.parameters()).data  # 当前权重θ
  
    # 2. 计算梯度并裁剪
    grads = torch.autograd.grad(loss, self.model.parameters())
    clip_coef = _clip(grads, self.network_clip)
  
    # 3. 计算更新量：∇θL_train + λθ
    dtheta = _concat(grads).data + self.network_weight_decay * theta
  
    # 4. 构建虚拟模型：θ' = θ - η(∇θL_train + λθ)
    unrolled_model = self._construct_model_from_theta(theta - eta * dtheta)
    return unrolled_model, clip_coef
```

###### 二阶优化 `_backward_step_unrolled`

```python
def _backward_step_unrolled(self, hidden_train, input_train, target_train,
                           hidden_valid, input_valid, target_valid, eta):
    # 1. 计算虚拟模型 w'
    unrolled_model, clip_coef = self._compute_unrolled_model(...)
  
    # 2. 在验证集计算损失：L_val(w', α)
    unrolled_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)
    unrolled_loss.backward()  # 计算梯度
  
    # 3. 获取架构梯度：∇αL_val(w', α)
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
  
    # 4. 获取权重梯度：∇w'L_val(w', α)
    dtheta = [v.grad for v in unrolled_model.parameters()]
    _clip(dtheta, self.network_clip)  # 裁剪梯度
  
    # 5. 近似海森矩阵向量积
    vector = [dt.data for dt in dtheta]
    implicit_grads = self._hessian_vector_product(vector, ...)
  
    # 6. 修正架构梯度：{∇}_{α} \boldsymbol{L}_{val}(\omega ^{*}(\alpha ),\alpha )

≈{∇}_{α} \boldsymbol{L}_{val}(\omega-ξ{∇}_{\omega}\boldsymbol{L}_{train}(\omega,\alpha), \alpha)
    for g, ig in zip(dalpha, implicit_grads):
        g.data.sub_(eta * clip_coef, ig.data)
  
    # 7. 更新当前模型的架构梯度
    for v, g in zip(self.model.arch_parameters(), dalpha):
        if v.grad is None:
            v.grad = Variable(g.data)
        else:
            v.grad.data.copy_(g.data)
    return hidden_next
```

###### 模型重建 `_construct_model_from_theta`

从参数向量重构模型实例

```python
def _construct_model_from_theta(self, theta):
    model_new = self.model.new()  # 创建新模型实例
    model_dict = self.model.state_dict()
  
    # 从theta向量重建参数
    params, offset = {}, 0
    for k, v in self.model.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset:offset+v_length].view(v.size())
        offset += v_length
  
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()
```

###### 执行流程

```python
def step(self, hidden_train, input_train, target_train,
        hidden_valid, input_valid, target_valid,
        network_optimizer, unrolled):
    eta = network_optimizer.param_groups[0]['lr']  # 获取学习率
  
    self.optimizer.zero_grad()  # 清空α的梯度
  
    if unrolled:  # 二阶优化
        self._backward_step_unrolled(...)
    else:        # 一阶优化
        self._backward_step(...)
  
    self.optimizer.step()  # 更新架构参数α
    return hidden_next
```
