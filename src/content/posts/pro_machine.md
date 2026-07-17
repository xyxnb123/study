---
title: "机器学习建模细节：数据准备、特征放缩、交叉验证、参数优化"
published: 2026-07-17
pinned: false
description: 详解机器学习完整建模落地流程，从零拆解数据清洗与预处理、特征标准化 / 归一化放缩、多方式交叉验证、模型超参数优化四大核心环节，梳理建模关键细节、常见踩坑点与实操技巧，适配入门学习与项目实战参考。
tags: [机器学习, 交叉验证, 参数优化, 特征放缩, 数据准备]
category: 机器学习
draft: false
# image: ./images/firefly2.avif
---

# 机器学习建模细节：数据准备、特征放缩、交叉验证、参数优化
## 一、数据准备：数据清洗与数据集划分）
### 1. 数据集划分 Train/Val/Test Split
#### 1.1 三集合分工
完整建模流程需要把原始数据拆分为**训练集、验证集、测试集**三份，各司其职：
1. **训练集（Train）**：模型学习数据分布、拟合权重参数，占比最大，通常70%~80%；
2. **验证集（Val）**：训练过程中观察模型效果，用来筛选超参数、提前判断过拟合；
3. **测试集（Test）**：全程只使用一次，作为模型上线前最终泛化能力的客观打分，绝对不能参与训练、调参、预处理。

#### 1.2 核心铁律：先划分，后归一化
这是机器学习最基础也最容易踩坑的数据泄露关键点：
- 错误做法：先对全量数据计算均值、方差、最大最小值完成缩放，再划分数据集；
- 问题：测试集的分布信息提前流入缩放参数，模型在测试集上分数虚高，上线真实数据效果暴跌；
- 正确流程：原始数据 → 划分 Train/Val/Test → 仅用训练集计算缩放统计量（均值、方差、四分位数等）→ 用训练集算出的参数同步转换验证集、测试集。

#### 1.3 划分比例参考
- 中小数据集：7:1:2（训练70%、验证10%、测试20%）
- 大数据集（十万级以上样本）：95:2.5:2.5，海量数据可进一步压缩验证、测试集占比
- 分类任务注意：划分时需保证各集合类别分布和原数据一致，优先使用分层划分。

#### 1.4 补充实操要点
1. 划分前必须打乱数据顺序，避免时序、排序带来的分布偏移；
2. 时序数据（股票、流量、日志）禁止随机打乱，按时间切分，防止未来信息泄露；
3. 测试集全程封存，网格搜索、交叉验证都只能使用训练+验证集，训练完成后再跑一次测试集输出最终结果。

#### 1.4 代码调用
&emsp;&emsp;使用 `sklearn.model_selection.train_test_split` 分层拆分，先拆测试集，再从剩余训练数据拆分验证集，严格分层保证类别均衡。

函数原型：

```python
train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
```
- **`*arrays`**
  传入需要划分的数据集，如 X, y，支持多个数组同步切分，划分后顺序一一对应。

- **`test_size`**
  测试集占比 / 样本数量：
  - 浮点数 0~1：代表样本比例，如 0.2 表示 20% 为测试集；
  - 整数：代表测试集绝对样本数量；
  - 不填则默认 0.25。

- **`train_size`**
  训练集占比 / 样本数量，与 test_size 二选一即可，两者相加不能超过 1。

- **`random_state`**
  随机种子，传入固定数字（如 42）保证每次划分结果完全一致，实验可复现。

- **`shuffle=True`（默认开启）**
  是否打乱数据顺序：
  - 表格、图像静态数据：保持 True；
  - 时序数据（股票、时序流量）：必须设为 False，防止未来数据泄露。

- **`stratify`**
  分层抽样核心参数，传入标签 y：
  - 仅分类任务使用；
  - 保证训练、测试集中各类别的样本占比和原始数据集完全相同；
  - 样本不均衡数据集必加，避免某一类全部进入训练 / 测试集。
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据集
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# 第一步：先拆分出测试集20%，分层抽样stratify=y保证类别分布
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 第二步：从训练验证集中再拆分验证集，总比例训练70%、验证10%
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val)
```
### 2. 缺失值填充 Imputation

#### 2.1 三集合分工缺失值三大产生机制（选填充方法的核心依据）
&emsp;&emsp;填充方案不能一刀切，先分清缺失类型：
- MCAR 完全随机缺失
缺失与自身、其他特征无关联，例如设备随机故障丢数据。

&emsp;&emsp;处理：少量缺失可直接删除，简单填充影响较小。
- MAR 随机缺失（业务最常见）缺失由其他特征决定，例如低收入用户不愿填写年收入。

&emsp;&emsp;处理：不能直接删样本，会造成样本分布偏移，推荐填充。
- MNAR 非随机缺失缺失和字段自身数值相关，例如高收入人群刻意隐藏收入。

&emsp;&emsp;处理：简单填充极易失真，优先通过业务补全原始数据。

#### 2.2 检测数据集缺失值
```Python
import pandas as pd
import missingno as msno
import numpy as np
# 读取数据
df = pd.read_csv("data.csv")

# 每列缺失总数
print(df.isnull().sum())

# 每列缺失占比（百分比）
missing_rate = df.isnull().sum() / len(df) * 100
print("缺失占比(%)：\n", missing_rate)
# 缺失矩阵图
msno.matrix(df)
# 缺失柱状图
msno.bar(df)
```
**缺失占比处理参考标准**
- 缺失率 ＜ 5%：完全随机缺失可直接删除样本
- 5% ≤ 缺失率 ≤ 30%：推荐填充处理
- 缺失率 ＞ 50%：字段有效信息过少，建议直接丢弃该列

#### 2.3 基础缺失值填充

（1）直接删除缺失数据：缺失占比极低、MCAR 完全随机缺失
```Python
# 删除存在空值的行
df_drop = df.dropna(axis=0)
# 删除整列全部为空的字段
df_drop_col = df.dropna(axis=1, how="all")
```
缺点：丢失样本，小数据集慎用，易造成数据量不足。

（2）常数填充：缺失本身具备业务含义（未填写、无采集记录）数值填固定值、分类特征填「未知」标签
```Python
# 数值统一填充0
df["score"] = df["score"].fillna(0)
# 分类字段填充未知
df["gender"] = df["gender"].fillna("未知")
```
缺点：强行修改原始分布，会引入偏差。

（3） 数值特征：均值 / 中位数填充
```Python
# 均值填充（无异常值、分布均匀数据）
df["income"] = df["income"].fillna(df["income"].mean())
# 中位数填充（存在极端离群值，推荐）
df["income"] = df["income"].fillna(df["income"].median())
```
（4）分类特征：众数填充，离散标签专用
```Python
mode_val = df["brand"].mode()[0]
df["brand"] = df["brand"].fillna(mode_val)
```
（5）进阶填充：时序插值（时间序列专用）均值填充会抹平时序趋势，传感器、流量、股票数据优先插值：
```Python
# 前向填充（用上一条数据填充空值）
df["flow"] = df["flow"].fillna(method="ffill")
# 后向填充（用下一条数据填充空值）
df["flow"] = df["flow"].fillna(method="bfill")
# 线性插值，还原连续变化趋势
df["flow"] = df["flow"].interpolate(method="linear")
```
（6）模型填充（机器学习，如：KNN 近邻填充，IterativeImputer 迭代多重填充）

**各类填充方法优缺点对比**
| 填充方法 | 优点 | 缺点 | 适用场景 |
| ---- | ---- | ---- | ---- |
| 删除法 | 无人工篡改数据分布 | 损失样本量 | 缺失极少、完全随机缺失 |
| 常数填充 | 逻辑简单、计算快 | 扭曲原始数据分布 | 缺失有业务含义的字段 |
| 均值填充 | 运算速度快 | 压缩方差、忽略极值 | 无异常值的平稳数值特征 |
| 中位数填充 | 抗异常值干扰 | 无法捕捉数据趋势 | 存在离群值的数值列 |
| 线性插值 | 保留时序变化规律 | 仅支持有序连续数据 | 时序、传感器连续数据 |
| KNN 填充 | 利用特征关联信息 | 大数据运行缓慢 | 中小数据集、特征相关性高 |
| 迭代模型填充 | 填充精度最高 | 计算开销大 | 机器学习建模、数据竞赛 |

### 3. 类别特征编码 One-Hot / Label Encoding
&emsp;&emsp;原始数据中经常存在大量文本型分类特征，例如性别、城市、商品品类、学历等级等。机器学习模型无法直接识别字符串文本，必须将离散类别转换成数值格式，这个转换过程就叫类别特征编码。最常用的两种编码方案分别是 Label Encoding（标签编码）与 One-Hot Encoding（独热编码），二者适用场景完全不同，混用极易造成模型逻辑偏差。
## 3\.1 Label Encoding 标签编码

**3.1.1 原理**

将分类字段的不同类别，按照字典顺序映射为一组连续整数：`0,1,2,3...`。
示例：学历字段 `["大专","本科","硕士","博士"]`，编码后变为 `[0,1,2,3]`。

**3.1.2 适用场景**

**仅适用于有序分类特征（序数特征）**
特征本身存在天然大小、高低、先后层级关系：

- 学历：大专 \< 本科 \< 硕士 \< 博士

- 评分等级：差 \< 一般 \< 良好 \< 优秀

- 年龄段：青年、中年、老年

模型会自动识别数字间的大小关系，贴合业务逻辑。
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["education_encode"] = le.fit_transform(df["education"])
```

## 3\.2 One\-Hot Encoding 独热编码

**3.2.1 原理**

为每一个独立类别新建一列，当前样本属于该类别则标记为 1，其余列全部为 0。
示例：城市 `["北京","上海","广州"]`，会生成三列：city\_北京、city\_上海、city\_广州。
北京样本：`[1,0,0]`；上海样本：`[0,1,0]`。

**3.2.2 适用场景**

**无序分类特征（名义特征）**
类别之间无大小、层级、先后之分：

- 城市、省份、商品品类、性别、品牌、用户 ID 等。
独热编码消除了类别之间虚假大小关系，所有类别权重平等，是无序分类的标准处理方案。

**3.2.3 补充关键说明**

独热编码输出全部是 0/1 二元稀疏特征，数值仅代表是否属于该类别，不存在数值尺度差异，**不需要额外做标准化、归一化缩放**，直接送入模型即可。多重共线性：N 个类别会产生 N 列，建模时建议删除其中一列，避免特征冗余。
```python
# pandas快速独热编码
df_onehot = pd.get_dummies(df["city"], prefix="city")
df = pd.concat([df, df_onehot], axis=1)
```
## 3\.3 两种编码核心对比总结

|编码方式|适用特征|输出形式|优缺点|
|---|---|---|---|
|Label Encoding|有序分类（学历、等级）|单个连续数字列|维度不膨胀；不可用于无序特征|
|One\-Hot Encoding|无序分类（城市、品类）|多列 0/1 二元特征|无虚假大小关系；高基数特征易维度爆炸|

##### 代码示例
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["level_encode"] = le.fit_transform(df["level"])
```
## 二、特征缩放：标准化、归一化与稳健缩放方案选型
&emsp;&emsp;在完成缺失值填充、类别特征编码后，数据预处理的核心下一步就是**特征缩放**。原始数据集的数值特征往往量纲差异极大，比如年龄（0\-100）、收入（0\-1000000）、消费评分（0\-10）。

&emsp;&emsp;如果不做缩放，数值量级更大的特征会直接主导模型训练，弱化小量级特征的权重，导致模型收敛缓慢、拟合偏差、距离类算法完全失效。特征缩放的核心目的就是**统一特征量纲、平衡特征权重、提升模型训练效率与精度**。

&emsp;&emsp;日常数据建模中，最常用的两种缩放方案为 **Z\-score 标准化** 和 **MinMax 归一化**，二者原理、特性、适用场景差异极大，需要根据数据分布和算法类型精准选型。

### 1\. Z\-score 标准化（Standardization）

**1.1 核心原理**

基于特征的均值和标准差进行变换，将数据转换为**均值为0、标准差为1**的标准正态分布，公式：$z = \frac{x-\mu}{\sigma}$。变换后数据无固定取值区间，可正可负。

#### 适用场景

- 数据近似服从**正态分布**的连续特征

- 无固定取值边界、数值波动范围较大的特征（如薪资、身高、温度、流水数据）

- 适配线性回归、逻辑回归、PCA降维、神经网络等多数主流算法

**1.2 特性与优缺点**

优点：保留了原始数据的离散程度和分布趋势，适配绝大多数常规建模场景，是工业建模默认首选方案。

缺点：**对极端异常值高度敏感**，异常值会大幅偏移均值和标准差，导致缩放后的数据失真。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["value_std"] = scaler.fit_transform(df[["value"]])
```

### 2\. MinMax 归一化（Normalization）

**2.1 核心原理**

基于特征的最大值和最小值做线性压缩，将所有数据强制映射到 **\[0,1\]** 固定区间，公式：$x_{new} = \frac{x-x_{min}}{x_{max}-x_{min}}$。

**2.2 适用场景**

- 有明确取值边界的特征数据（图像像素、评分、比例、概率数据）

- **距离敏感型算法**：KNN、SVM、聚类算法、梯度下降类算法

- 深度学习、神经网络模型输入数据

**2.3 特性与优缺点**

优点：统一固定区间，彻底消除量纲影响，能够**大幅加速梯度下降收敛速度**，提升迭代效率。

缺点：极度依赖最大/最小值，对异常值极其敏感，一旦存在极端值，会压缩正常数据的分布区间，丢失有效特征信息。

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df["value_norm"] = scaler.fit_transform(df[["value"]])
```

### 3\. 拓展：稳健缩放（RobustScaler）

针对上述两种方法对异常值敏感的痛点，补充工业常用的**稳健缩放**方案，专门适配含大量异常值的数据集。

原理基于中位数和四分位数，避开极值干扰，抗干扰能力极强。适合脏数据、工业传感器数据、金融流水等噪声大的场景。

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df["value_robust"] = scaler.fit_transform(df[["value"]])
```

### 4\. 三种缩放方案核心选型总结

|缩放方式|区间|抗异常值|核心适用场景|
|---|---|---|---|
|Z\-score 标准化|无固定区间|弱|正态分布数据、常规机器学习建模、PCA降维|
|MinMax 归一化|\[0,1\]|极弱|有边界数据、KNN/SVM、深度学习、梯度优化场景|
|Robust 稳健缩放|无固定区间|极强|含异常值、噪声大的工业/金融/传感器数据|

# 三、模型评估：交叉验证策略与任务评价指标

&emsp;&emsp;在模型训练与调参过程中，单纯依靠单次训练集/测试集划分容易受数据随机性影响，导致结果不可靠、泛化能力误判。同时不同任务需要对应专属评价指标，不能仅凭单一分数判断模型好坏。本节介绍机器学习建模中最通用的**交叉验证策略**与**分类/回归标准评价指标**，是建模对比、超参搜索、模型选型的核心依据。

### 1\. K折交叉验证（K\-Fold CV）

**1.1 核心原理**

将完整数据集均匀划分为 K 个子集，每一轮轮流选取其中 1 份作为验证集，剩余 K\-1 份作为训练集，循环迭代 K 次，最终取 K 轮指标平均值作为模型最终性能分数。

**1.2 核心作用**

- 消除单次划分的随机性，避免“运气好/运气差”的测试结果

- 充分利用全部数据，小数据集尤其提升评估可靠性

- 得到更稳定、客观的**泛化能力评估**

**1.3 适用场景**

通用所有机器学习任务（回归、普通分类），是默认首选验证方式。
```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)
```
### 2\. 分层K折交叉验证（Stratified K\-Fold）

**1.1 核心原理**

&emsp;&emsp;在普通 KFold 的基础上，**保证每一份数据的类别分布比例与原数据集完全一致**，避免某一折样本类别失衡、数据分布偏移。

**1.2 适配场景**

- **仅用于分类任务**

- 数据集类别不均衡场景（正负样本比例差距大）

**1.3 核心优势**

每一轮训练、验证集都保持相同数据分布，有效避免因抽样不均导致的评估虚高或虚低，是**分类任务标准验证方案**。
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```
### 3\. 建模评价指标（Metrics）

&emsp;&emsp;模型训练、网格搜索、超参数优化都需要固定评价指标作为目标，分类、回归任务指标体系完全不同，需要严格区分使用。

## 3. 建模评价指标 Metrics

**3.1 分类任务指标**
适用于二分类、多分类场景，常用四大核心指标：**准确率、精确率、召回率、F1分数**。
混淆矩阵基础定义：TP=真正例，FP=假正例，TN=真负例，FN=假负例

- **准确率 Accuracy**
公式：
$$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$$
整体预测正确样本占全部样本的比例，适合样本均衡场景；样本不均衡时该指标容易产生误导，无法真实反映模型性能。

- **精确率 Precision**
公式：
$$Precision = \frac{TP}{TP+FP}$$
所有被预测为正的样本中真实为正的比例，侧重减少误判。

- **召回率 Recall**
公式：
$$Recall = \frac{TP}{TP+FN}$$
所有真实正样本中被成功预测出来的比例，侧重减少漏判。

- **F1-Score**
公式：
$$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$
精确率与召回率的调和平均，均衡两项指标，综合衡量模型整体分类能力。

**3.2 回归任务指标**
适用于连续值预测任务，常用核心指标：**MSE、RMSE、MAE、R²**。
$y_i$ 为真实标签，$\hat{y_i}$ 为预测值，$n$ 为样本总量，$\bar{y}$ 为真实标签均值。

- **MSE（均方误差）**
公式：
$$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y_i})^2$$
衡量预测值与真实值误差平方的平均值，对大误差惩罚更强，数值越小模型拟合越好。

- **RMSE（均方根误差）**
公式：
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y_i})^2}$$
MSE开平方，误差单位和原始数据保持一致，便于业务解读。

- **MAE（平均绝对误差）**
公式：
$$MAE = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y_i}|$$
误差绝对值平均，受极端异常值影响更小。

- **R² 决定系数**
公式：
$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y_i})^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
衡量特征对目标变量的解释能力，取值区间 $(-\infty,1]$；越接近1拟合效果越好，小于0代表模型效果极差。

**3.3 指标与调参的关联逻辑**

&emsp;&emsp;网格搜索、随机搜索、贝叶斯调参等超参数优化过程，**都是以评价指标作为优化目标**：分类最大化 F1/准确率，回归最小化 MSE、最大化 R²，以此筛选最优模型参数组合。

## 四、超参数调优：网格、随机与贝叶斯优化对比
### 超参数调优两大实现思路
超参数寻找分为「简单采样遍历」和「优化算法迭代寻优」两类，二者本质完全不同：
1. 采样遍历方式（网格搜索、随机搜索）
仅预设参数范围，通过枚举/随机抽取组合训练模型，不会记录、学习历史实验结果，不存在自主寻优逻辑，不属于优化算法。
优点：简单易实现；缺点：算力浪费严重，高维参数空间效率极低。

2. 优化算法迭代寻优（贝叶斯优化、遗传算法、PSO粒子群等）
依靠完整优化逻辑自动搜索最优参数，属于专业优化算法。以贝叶斯优化为代表，通过代理模型拟合参数与模型指标的关系，使用采集函数预判下一组潜力参数，迭代收敛至全局最优。
同等算力下寻优效果远优于暴力采样，是工业高精度调参首选。

下面列举几个常用的方法。

### 1. 网格搜索 GridSearchCV

**1.1 核心原理**
预先设定每个超参数的候选取值列表，对**所有参数组合进行穷举遍历**，每一组参数都会执行交叉验证，最终返回验证指标最优的参数搭配。

**1.2适用场景**
- 超参数数量少、参数维度低
- 每个参数候选范围有限、取值数量少
- 算力充足，需要遍历全部组合保证不漏最优解

**1.3 优缺点**
- 优点：遍历全部组合，一定能找到预设范围内最优参数；操作简单易上手
- 缺点：参数一多组合呈指数爆炸，计算成本极高，高维参数空间完全不适用

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators": [100, 200], "max_depth": [3, 5, 7]}
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
```

### 2\. 随机搜索 RandomizedSearchCV

**2.1 核心原理**

不再遍历全部组合，在完整参数空间中随机抽取固定数量的参数组合训练，通过有限采样寻找较优参数。

**1.2适用场景**

- 超参数维度较多、参数取值范围宽泛

- 算力有限，无法承受网格搜索巨大计算量

- 前期粗筛大范围参数，快速锁定优质参数区间

**2.3 优缺点**

- 优点：计算速度快，更容易跳出局部最优，适合大范围粗调

- 缺点：属于随机采样，大概率错过全局最优参数组合

```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

param_dist = {"n_estimators": stats.randint(50, 500), "max_depth": stats.randint(2, 12)}
rand_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=30, cv=5)
rand_search.fit(X_train, y_train)
```

### 3\. 贝叶斯优化 Bayesian Optimization（Optuna）

**3.1 核心原理**

区别于网格、随机的无记忆搜索，贝叶斯优化会记录每一组参数对应的模型指标，基于历史结果构建概率模型，**迭代预测最有可能产出高分的参数区域**，优先在潜力高的参数位置采样，智能缩小搜索范围。

程实战、竞赛主流进阶优化方案。
**3.2适用场景**

- 参数维度高、模型训练耗时大（神经网络、集成树大模型）

- 追求高精度调参，算力有限但希望用少量迭代拿到最优参数

- 线上工程、模型竞赛高精度需求场景

**3.3 优缺点**

- 优点：利用历史搜索信息，收敛速度远快于网格 / 随机搜索，少量迭代即可找到优质参数

- 缺点：逻辑相对复杂，入门成本略高；依赖第三方库（Optuna/GPyOpt）

```python
import optuna
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 12)
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_params)
```

### 三种调参方案对比

|调优方法|搜索逻辑|计算速度|适用场景|
|---|---|---|---|
|网格搜索 GridSearchCV|穷举全部参数组合|最慢|少量超参数、候选值少、低维参数空间|
|随机搜索 RandomizedSearchCV|随机采样参数组合|中等|参数较多、大范围粗筛、算力一般|
|贝叶斯优化 Optuna|基于历史结果智能预测采样|最快|高维参数、训练耗时模型、高精度调参|



---


