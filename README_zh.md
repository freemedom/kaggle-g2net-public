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

