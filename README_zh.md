## Kaggle G2Net 公共方案代码概览（中文）

本目录包含 2021 Kaggle “Gravitational Wave Detection” 比赛的 2nd place 方案代码与配置，核心流程是将三通道引力波时序信号转换为多种时频图或一维卷积特征，训练多模型集成并使用伪标签与 stacking 提升成绩。

### 主要文件与职责
- `README.md`：官方英文使用说明与 20 个实验列表。
- `configs.py`：全部实验/伪标签配置类，定义数据路径、拆分、模型、优化器、调度器、数据增强与回调等。
- `train.py` / `train_pseudo.py`：按配置训练或推理（支持 TTA、只跑单折、断点续训等），输出 oof/predictions/checkpoints 与日志。
- `prep_data.py`：根据 Kaggle 原始结构生成 `input/train.csv`、`input/test.csv`，可选生成 waveform cache。
- `datasets.py`：`G2NetDataset` 读取 .npy 波形，支持缓存、mixup、伪标签与二次增强。
- `transforms.py`：音频增强库（归一化、带通滤波、噪声注入、反转、随机裁剪/重采样、频域操作等）。
- `architectures.py`：时频图 CNN 与多实例模型封装（SpectroCNN、MultiInstanceSCNN），可组合 CWT/CQT/学习型 WaveNet/CNN 频谱、timm 主干与自定义分类头、mixup/cutmix。
- `models1d_pytorch/`：1D 模型实现（ResNet1d、DenseNet1d、WaveNet1d 等）。
- 其它：`training_extras.py`（Mixup 钩子、TTA loader）、`loss_functions.py`（带标签平滑的 BCE/Focal）、`metrics.py`（AUC）、`utils.py`（配置打印与通知）。

### 数据准备
1) 下载比赛数据至 `input/`。  
2) 运行 `python prep_data.py` 生成 `train.csv`/`test.csv`（可加 `--cache` 生成波形缓存）。  
3) 若生成缓存，将路径写入 `configs.py` 对应 `train_cache`/`test_cache`。

### 训练与推理
- 单模型：`python train.py --config {配置类名} [--tta --progress_bar --limit_fold k --inference]`
- 伪标签：`python train_pseudo.py --config {Pseudo*}`（依赖前置实验输出）。
- 输出位于 `results/{name}/`，包含 `fold*.pt`、`outoffolds.npy`、`predictions.npy` 等。
- Stacking：运行 `g2net-submission.ipynb` 汇总 20 个模型预测生成提交。

### 配置与实验要点（见 `configs.py`）
- 基线：CQT 频谱 + EfficientNet-B7；5 折 StratifiedKFold；CosineWarmRestarts 调度；AMP。
- Spectrogram 变体：Morlet CWT、WaveNet/CNN 学习型频谱、多频谱拼接、多 patch (MultiInstance)。
- 主干多样：EfficientNet 系列、DenseNet、ResNet200d、XCiT、EffNetV2 等，自定义池化/注意力/分类头（GeM、TripletAttention、PositionalEncoding、ManifoldMixup）。
- 1D 序列线：ResNet1d、DenseNet1d、WaveNet1d 配合带通+归一化增强。
- 伪标签：`Pseudo*` / `PseudoSeq*` 类从对应实验的 `predictions.npy` 读伪标签再训练。

### 数据增强与正则
- 波形级：Normalize、BandPass、Gaussian/Pink/Brown 噪声注入、FlipWave、随机裁剪/重采样、色噪声等。
- 训练钩子：`MixupTrain`/`MixupTrain2` 支持输入/特征层 mixup、正负样本分组混合，可与 cutmix 搭配。
- TTA：推理阶段可对波形翻转并平均预测。

### 快速开始示例
```bash
conda env create -n kumaconda -f=environment.yaml
conda activate kumaconda
python prep_data.py --cache        # 可选
python train.py --config Nspec16   # 训练 CWT + EfficientNet-B2 模型
python train.py --config Nspec16 --tta --inference  # 仅推理 + TTA
```

### 生成物
- 每个实验目录：`fold*.pt` 权重、`outoffolds.npy`、`predictions.npy`、训练日志。  
- 最终提交：在 notebook 中 stacking 后导出 `results/submission.csv`。











好的，MBConv（Mobile Inverted Bottleneck Convolution，移动反向瓶颈卷积）是 **EfficientNet** 和 **MobileNetV2** 等高效模型架构中的核心构建块。它之所以高效，是因为它成功地平衡了网络的**深度**、**宽度**和**分辨率**，并在保证高性能的同时显著减少了计算量和参数数量。

`MBConvX` 中的 **X** 代表 **扩张因子 (Expansion Factor)**，决定了中间层通道数的膨胀倍数。

---

## 🔎 MBConv 详解：工作原理

MBConv 块得名于它的**反向瓶颈结构**和对**深度可分离卷积**的使用。

### 1. 反向瓶颈结构 (Inverted Bottleneck)

传统的瓶颈结构（如 ResNet）是：**宽 $\rightarrow$ 窄 $\rightarrow$ 宽**。它首先使用 $1 \times 1$ 卷积减少通道数（压缩信息），再进行 $3 \times 3$ 卷积处理，最后再用 $1 \times 1$ 卷积恢复通道数。

MBConv 采用**反向**结构：**窄 $\rightarrow$ 宽 $\rightarrow$ 窄**。

* **输入（窄）**: 较低维度的输入特征。
* **中间（宽）**: 使用 $1 \times 1$ 卷积将通道数**扩张** $X$ 倍（通常 $X=6$）。
* **输出（窄）**: 使用 $1 \times 1$ 卷积将通道数**投影**回较低维度。

这种设计理念是基于一个洞察：如果你使用深度可分离卷积 (Depthwise Convolution) 来处理特征，那么在**低维空间**中的信息是紧凑的，应该先扩展到**高维空间**，让 $3 \times 3$ 卷积在更丰富的特征上学习，然后再压缩回低维输出。

### 2. 核心步骤与结构

一个 MBConvX 块通常由以下三个主要部分组成，并且包含残差连接和 Squeeze-and-Excitation (SE) 模块：

#### 步骤 1: 通道扩张（$1 \times 1$ Conv）
* **作用**: 将输入特征的通道数从 $C_{in}$ 扩展到 $C_{in} \times X$。
* **细节**: 使用 $1 \times 1$ 卷积层。
* **激活函数**: **Swish**（或称 SILU）。

#### 步骤 2: 深度可分离卷积 (Depthwise Conv)
* **作用**: 在扩张后的高维特征上进行空间特征提取。
* **细节**: 使用 $3 \times 3$ 或 $5 \times 5$ 深度卷积。深度卷积的计算量远小于标准卷积。
    * 在这一步之后，通常会插入 **Squeeze-and-Excitation (SE) 模块**，以自适应地重新校准通道权重，增强模型的表达能力。
* **激活函数**: **Swish**。

#### 步骤 3: 投影（$1 \times 1$ Conv）
* **作用**: 将通道数从 $C_{in} \times X$ 投影回输出通道数 $C_{out}$。
* **细节**: 使用 $1 \times 1$ 卷积层。
* **激活函数**: **无**（通常不使用激活函数，保持线性输出，以保证残差连接的有效性）。

#### 步骤 4: 残差连接 (Residual Connection)
* **条件**: 只有当**输入通道数**和**输出通道数**相同时，并且**步长（Stride）为 1** 时，才会添加残差连接（即输入直接加到投影输出上）。



---

## 🔢 MBConvX 中的 “X”：扩张因子

**扩张因子 X**（Expansion Factor）是决定 MBConv 块计算成本和信息容量的关键参数。

* **定义**: $X$ 是指中间扩张层通道数相对于输入通道数的倍数。
    $$\text{中间通道数} = \text{输入通道数} \times X$$
* **高效Net-B0 中的 X**：在 EfficientNet-B0 中，除第一层 MBConv ( $X=1$ ) 外，所有后续的 MBConv 块都使用 $X=6$。
    * **MBConv6** 意味着输入通道数被扩展了 6 倍。
* **影响**: $X$ 越大，中间层的通道越多，模型的**表达能力越强**，但**计算量**也会相应增加。

MBConv 结构之所以高效，正是因为它通过**扩张因子 $X$** 在通道维度上创造了一个高维空间来丰富特征，同时利用**深度可分离卷积**在空间维度上保持了极低的计算开销。