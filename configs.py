"""
Kaggle G2Net 引力波检测竞赛 - 模型训练配置文件

本文件包含所有实验的配置类，用于定义模型架构、训练参数、数据增强策略等。
每个配置类继承自基础配置类，通过修改特定参数来创建不同的实验变体。

主要配置类型：
1. Baseline: 基础配置，使用CQT时频变换和EfficientNet
2. Nspec系列: 使用连续小波变换(CWT)的配置
3. Seq系列: 使用1D序列模型（ResNet1d, DenseNet1d, WaveNet1d）的配置
4. Pseudo系列: 使用伪标签训练的配置
5. MultiInstance: 多实例学习配置
"""

from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from nnAudio.Spectrogram import CQT  # 常数Q变换（时频分析）
from cwt_pytorch import ComplexMorletCWT  # 复Morlet小波变换

from kuma_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot, SaveEveryEpoch, SaveAllSnapshots)
from kuma_utils.torch.hooks import TrainHook

from datasets import G2NetDataset
from architectures import SpectroCNN, MultiInstanceSCNN
from models1d_pytorch import *
from loss_functions import BCEWithLogitsLoss
from metrics import AUC
from transforms import *


# 数据目录配置
INPUT_DIR = Path('input/').expanduser()

# 硬件配置字典：用于根据硬件资源调整训练参数
# 格式：(CPU核心数, RAM大小(GB), GPU数量, GPU显存总量(GB))
HW_CFG = {
    'RTX3090': (16, 128, 1, 24), # CPU cores, RAM amount, GPU count, GPU RAM total
    'A100': (9, 60, 1, 40), 
}


class Baseline:
    """
    基础配置类 - 所有其他配置类的父类
    
    使用CQT（常数Q变换）将时域信号转换为时频图，然后用EfficientNet-B7进行图像分类。
    这是最简单的baseline配置，没有数据增强。
    """
    name = 'baseline'  # 实验名称
    seed = 2021  # 随机种子，确保可复现性
    train_path = INPUT_DIR/'train.csv'  # 训练集CSV路径
    test_path = INPUT_DIR/'test.csv'   # 测试集CSV路径
    train_cache = None  # 训练集缓存路径（可选，用于加速数据加载）
    test_cache = None   # 测试集缓存路径（可选）
    
    # 交叉验证配置
    cv = 5  # 5折交叉验证
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)  # 分层K折，保持类别分布
    
    # 数据集配置
    dataset = G2NetDataset  # 数据集类
    dataset_params = dict()  # 数据集额外参数

    # 模型配置
    model = SpectroCNN  # 频谱图CNN模型（将时域信号转为频谱图后分类）
    model_params = dict(
        model_name='tf_efficientnet_b7',  # TensorFlow版本的EfficientNet-B7（ImageNet预训练）
        pretrained=True,  # 使用预训练权重
        num_classes=1,  # 二分类任务（输出1个值）
        spectrogram=CQT,  # 使用CQT（常数Q变换）作为时频变换方法
        spec_params=dict(
            sr=2048,      # 采样率：2048 Hz（引力波数据采样率）
            fmin=20,      # 最小频率：20 Hz
            fmax=1024,    # 最大频率：1024 Hz
            hop_length=64 # 跳跃长度：64（控制时间分辨率）
        ),
    )
    weight_path = None  # 预训练权重路径（None表示从头训练）
    
    # 训练超参数
    num_epochs = 5  # 训练轮数
    batch_size = 64  # 批次大小
    
    # 优化器配置
    optimizer = optim.Adam  # Adam优化器
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)  # 学习率2e-4，权重衰减1e-6
    
    # 学习率调度器配置
    scheduler = CosineAnnealingWarmRestarts  # 余弦退火带重启
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)  # 初始周期5，最小学习率1e-6
    scheduler_target = None  # 调度器监控指标（None表示监控loss）
    batch_scheduler = False  # 是否每个batch更新学习率（False表示每个epoch更新）
    
    # 损失函数和评估指标
    criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失（带sigmoid）
    eval_metric = AUC().torch  # 评估指标：AUC（ROC曲线下面积）
    monitor_metrics = []  # 额外监控指标列表
    
    # 训练技术配置
    amp = True  # 自动混合精度训练（FP16），加速训练并节省显存
    parallel = None  # 并行训练配置（None表示单GPU）
    deterministic = False  # 是否使用确定性算法（False表示允许非确定性操作以提升性能）
    clip_grad = 'value'  # 梯度裁剪方式：'value'表示按值裁剪
    max_grad_norm = 10000  # 梯度裁剪阈值（非常大，实际不裁剪）
    
    # 训练钩子和回调函数
    hook = TrainHook()  # 训练钩子
    callbacks = [
        EarlyStopping(patience=5, maximize=True),  # 早停：5个epoch无提升则停止，监控指标越大越好
        SaveSnapshot()  # 保存模型快照
    ]

    # 数据增强配置
    transforms = dict(
        train=None,  # 训练集数据增强（None表示无增强）
        test=None,   # 测试集数据增强
        tta=None     # 测试时增强（Test Time Augmentation）
    )

    # 伪标签配置
    pseudo_labels = None  # 伪标签路径（None表示不使用伪标签）
    debug = False  # 调试模式（False表示正常训练）


class Resized08aug4(Baseline):
    """
    改进的基础配置 - 添加了数据增强和图像尺寸调整
    
    主要改进：
    - 使用更小的EfficientNet-B2（相比B7更快）
    - 添加高斯噪声增强（SNR 15-30 dB）
    - 将频谱图调整到256x512尺寸
    - 使用更小的hop_length=8（更高的时间分辨率）
    """
    name = 'resized_08_aug_4'
    model_params = dict(
        model_name='tf_efficientnet_b2',  # 更小的模型，训练更快
        pretrained=True,
        num_classes=1,
        spectrogram=CQT,
        spec_params=dict(
            sr=2048, 
            fmin=16,      # 降低最小频率到16 Hz
            fmax=1024, 
            hop_length=8  # 更小的跳跃长度，提高时间分辨率
        ),
        resize_img=(256, 512),  # 将频谱图调整到256x512
        upsample='bicubic'  # 使用双三次插值上采样
    )
    transforms = dict(
        train=Compose([
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.25)  # 25%概率添加高斯噪声
        ]),
        test=None,
        tta=None
    )
    dataset_params = dict(
        norm_factor=[4.61e-20, 4.23e-20, 1.11e-20]  # 三个探测器的归一化因子
    )
    num_epochs = 8  # 增加训练轮数
    scheduler_params = dict(T_0=8, T_mult=1, eta_min=1e-6)  # 调整调度器周期
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)  # 提高学习率到1e-3


class Nspec12(Resized08aug4):
    """
    使用连续小波变换(CWT)的配置 - Nspec12
    
    主要特点：
    - 使用ComplexMorletCWT替代CQT（小波变换对瞬态信号更敏感）
    - 添加带通滤波器（12-512 Hz）
    - 256个尺度的小波变换
    """
    name = 'nspec_12'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,  # 复Morlet小波变换（更适合引力波信号）
        spec_params=dict(
            fs=2048,         # 采样率
            lower_freq=16,   # 最低频率
            upper_freq=1024, # 最高频率
            wavelet_width=3, # 小波宽度参数
            stride=8,        # 时间步长
            n_scales=256     # 小波尺度数量（频率分辨率）
        ),
        resize_img=(256, 512),
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            BandPass(lower=12, upper=512),  # 带通滤波：保留12-512 Hz频率范围
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),  # 50%概率添加噪声
        ]),
        test=BandPass(lower=12, upper=512),  # 测试时也应用带通滤波
        tta=BandPass(lower=12, upper=512)     # TTA时也应用带通滤波
    )


class Nspec12arch0(Nspec12):
    """
    Nspec12的架构变体 - 使用DenseNet201作为backbone
    
    主要变化：
    - 将backbone从EfficientNet-B2改为DenseNet201
    - 使用更大的小波宽度（wavelet_width=8）
    - 适合需要更强特征提取能力的场景
    """
    name = 'nspec_12_arch_0'
    model_params = dict(
        model_name='densenet201',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=8,
                         stride=8,
                         n_scales=256),
        resize_img=(256, 512),
        upsample='bicubic'
    )
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Nspec16(Resized08aug4): 
    """
    使用CWT的高性能配置 - Nspec16
    
    主要特点：
    - 可训练的小波宽度（trainable_width=True），允许模型学习最优小波参数
    - 使用GeM池化（Generalized Mean Pooling）替代平均池化
    - 更高的时间分辨率（stride=4）
    - 图像尺寸128x1024（更宽的时间维度）
    """
    name = 'nspec_16'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(
            fs=2048, 
            lower_freq=16, 
            upper_freq=1024, 
            wavelet_width=8,      # 更大的小波宽度
            trainable_width=True, # 可训练的小波宽度（端到端优化）
            stride=4,             # 更小的步长，提高时间分辨率
            n_scales=128          # 128个尺度
        ),
        resize_img=(128, 1024),   # 更宽的时间维度（1024）
        custom_classifier='gem',  # GeM池化（Generalized Mean Pooling）
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),  # 归一化三个探测器
            BandPass(lower=12, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ])
    )
    dataset_params = dict()


class Nspec16spec13(Nspec16):
    """
    Nspec16的频谱参数变体 - 限制频率范围
    
    主要变化：
    - 降低最大频率到512 Hz（从1024 Hz）
    - 带通滤波器范围改为12-360 Hz（更窄的频率范围）
    - 使用4阶滤波器（order=4）提高滤波精度
    - 适合专注于低频引力波信号的场景
    """
    name = 'nspec_16_spec_13'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=512, 
                         wavelet_width=8,
                         trainable_width=True, 
                         stride=4,
                         n_scales=128),
        custom_classifier='gem', 
        upsample='bicubic',
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=360, order=4),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=360, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=360, order=4),
        ])
    )


class Nspec16arch17(Nspec16):
    """
    Nspec16的大模型变体 - 使用EfficientNet-B7-NS
    
    主要变化：
    - 使用更大的EfficientNet-B7-NS模型（Noisy Student预训练）
    - 不调整图像尺寸（resize_img=None），保持原始分辨率
    - 减小batch_size到32以适应更大的模型
    - 提高学习率到5e-4以加速大模型训练
    - 适合有充足计算资源的场景
    """
    name = 'nspec_16_arch_17'
    model_params = dict(
        model_name='tf_efficientnet_b7_ns',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=8,
                         trainable_width=True, 
                         stride=4,
                         n_scales=128),
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )
    batch_size = 32
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)


class Nspec21(Resized08aug4):
    """
    使用CWT的中等规模配置 - Nspec21
    
    主要特点：
    - 使用EfficientNet-B4-NS作为backbone（中等规模）
    - 256个小波尺度（更高的频率分辨率）
    - 图像尺寸256x1024（更宽的时间维度）
    - 可训练的小波宽度，允许端到端优化
    """
    name = 'nspec_21'
    model_params = dict(
        model_name='tf_efficientnet_b4_ns',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=8,
                         trainable_width=True, 
                         stride=4,
                         n_scales=256),
        resize_img=(256, 1024),
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ])
    )
    dataset_params = dict()


class Nspec22(Resized08aug4):
    """
    使用WaveNet作为时频变换的配置 - Nspec22
    
    主要特点：
    - 使用WaveNetSpectrogram（可学习的时频变换，而非固定的小波变换）
    - WaveNet可以学习最优的时频表示
    - 不调整图像尺寸（resize_img=None），保持原始分辨率
    """
    name = 'nspec_22'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=WaveNetSpectrogram,  # 使用WaveNet作为可学习的时频变换
        spec_params=dict(
            base_filters=128,      # 基础滤波器数量
            wave_layers=(10, 6, 2), # 各层的WaveNet块数量
            kernel_size=3,         # 卷积核大小
        ), 
        resize_img=None,  # 不调整尺寸，保持原始分辨率
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
    )
    dataset_params = dict()


class Nspec22aug1(Nspec22):
    """
    Nspec22的数据增强变体 - 添加波形翻转增强
    
    主要变化：
    - 移除WaveNet块（wave_block='none'），简化架构
    - 添加FlipWave增强（50%概率翻转波形）
    - FlipWave可以增加数据多样性，提升模型泛化能力
    - 适合需要更强数据增强的场景
    """
    name = 'nspec_22_aug_1'
    model_params = Nspec22.model_params.copy()
    model_params['spec_params'] =dict(
        wave_block='none',
        base_filters=128,
        wave_layers=(10, 6, 2),
        kernel_size=3,
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
            FlipWave(p=0.5)
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
    )


class Nspec22arch2(Nspec22): 
    """
    Nspec22aug1的架构变体 - 使用EfficientNet-B6-NS
    
    主要变化：
    - 将backbone从EfficientNet-B2升级到B6-NS
    - 继承Nspec22aug1的数据增强策略
    - 更大的模型容量，适合复杂模式识别
    """
    name = 'nspec_22_arch_2'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'tf_efficientnet_b6_ns'
    transforms = Nspec22aug1.transforms.copy()


class Nspec22arch6(Nspec22):
    """
    Nspec22aug1的架构变体 - 使用DenseNet201
    
    主要变化：
    - 使用DenseNet201作为backbone（密集连接架构）
    - 降低学习率到2e-4以适应DenseNet的训练特性
    - DenseNet通过特征重用提升参数效率
    """
    name = 'nspec_22_arch_6'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'densenet201'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Nspec22arch7(Nspec22):
    """
    Nspec22aug1的架构变体 - 使用EfficientNetV2-M
    
    主要变化：
    - 使用EfficientNetV2-M（改进的EfficientNet架构）
    - EfficientNetV2在训练速度和精度之间取得更好平衡
    - 降低学习率到2e-4
    """
    name = 'nspec_22_arch_7'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'tf_efficientnetv2_m'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Nspec22arch10(Nspec22):
    """
    Nspec22aug1的架构变体 - 使用ResNet200D
    
    主要变化：
    - 使用ResNet200D（深度残差网络，200层）
    - 提高学习率到5e-4以加速深度网络训练
    - ResNet200D适合需要极深网络的场景
    """
    name = 'nspec_22_arch_10'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'resnet200d'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)


class Nspec22arch12(Nspec22):
    """
    Nspec22aug1的架构变体 - 使用EfficientNetV2-L
    
    主要变化：
    - 使用EfficientNetV2-L（大型EfficientNetV2模型）
    - 减小batch_size到32以适应大模型
    - 降低学习率到2e-4
    - 适合有充足显存的场景
    """
    name = 'nspec_22_arch_12'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'tf_efficientnetv2_l'
    transforms = Nspec22aug1.transforms.copy()
    batch_size = 32
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Nspec23(Nspec22):
    """
    使用CNN频谱图的配置 - Nspec23
    
    主要特点：
    - 使用CNNSpectrogram替代WaveNet（可学习的CNN时频变换）
    - 使用多尺度卷积核（32, 16, 4）捕获不同时间尺度的特征
    - CNN频谱图比固定变换更灵活，可以学习最优表示
    """
    name = 'nspec_23'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=CNNSpectrogram,
        spec_params=dict(
            base_filters=128, 
            kernel_sizes=(32, 16, 4), 
        ),
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )


class Nspec23arch3(Nspec23):
    """
    Nspec23的架构和参数变体
    
    主要变化：
    - 使用EfficientNet-B6-NS作为backbone
    - 增大第一个卷积核到64（从32），捕获更长的时间模式
    - 继承Nspec22aug1的数据增强策略
    """
    name = 'nspec_23_arch_3'
    model_params = Nspec23.model_params.copy()
    model_params['spec_params'] = dict(
        base_filters=128, 
        kernel_sizes=(64, 16, 4), 
    )
    model_params['model_name'] = 'tf_efficientnet_b6_ns'
    transforms = Nspec22aug1.transforms.copy()


class Nspec23arch5(Nspec23):
    """
    Nspec23arch3的架构变体 - 使用EfficientNetV2-M
    
    主要变化：
    - 将backbone从EfficientNet-B6-NS改为EfficientNetV2-M
    - 提高学习率到5e-4
    - EfficientNetV2-M在精度和速度之间取得平衡
    """
    name = 'nspec_23_arch_5'
    model_params = Nspec23arch3.model_params.copy()
    model_params['model_name'] = 'tf_efficientnetv2_m'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)


class Nspec25(Nspec22):
    """
    WaveNet频谱图的改进配置 - Nspec25
    
    主要特点：
    - 使用更大的基础滤波器（256，从128增加）
    - 添加下采样（downsample=4）减少计算量
    - 移除WaveNet块（wave_block='none'），简化架构
    - 继承Nspec22aug1的数据增强策略
    """
    name = 'nspec_25'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=WaveNetSpectrogram,
        spec_params=dict(
            wave_block='none',
            base_filters=256,
            wave_layers=(10, 6, 2),
            downsample=4,
        ), 
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = Nspec22aug1.transforms.copy()


class Nspec25arch1(Nspec25):
    """
    Nspec25的架构变体 - 使用EfficientNet-B3-NS
    
    主要变化：
    - 将backbone从EfficientNet-B2升级到B3-NS
    - 保持Nspec25的所有其他配置不变
    """
    name = 'nspec_25_arch_1'
    model_params = Nspec25.model_params
    model_params['model_name'] = 'tf_efficientnet_b3_ns'


class Nspec30(Nspec22):
    """
    使用分离通道WaveNet的配置 - Nspec30
    
    主要特点：
    - 使用separate_channel=True，为每个输入通道单独处理
    - 更深的WaveNet层（10, 10, 10），每层10个块
    - 适合需要精细处理多通道信号的场景
    - 每个通道独立学习时频表示，增强模型表达能力
    """
    name = 'nspec_30'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=WaveNetSpectrogram,
        spec_params=dict(
            separate_channel=True,
            in_channels=1, 
            wave_block='none',
            base_filters=128,
            wave_layers=(10, 10, 10),
            kernel_size=3,
        ),
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = Nspec22aug1.transforms.copy()


class Nspec30arch2(Nspec30):
    """
    Nspec30的架构和参数变体
    
    主要变化：
    - 使用EfficientNet-B6-NS作为backbone
    - 减少WaveNet层数（8, 8, 8），每层8个块
    - 在模型容量和计算效率之间取得平衡
    """
    name = 'nspec_30_arch_2'
    model_params = Nspec30.model_params.copy()
    model_params['spec_params'] = dict(
        separate_channel=True,
        in_channels=1, 
        wave_block='none',
        base_filters=128,
        wave_layers=(8, 8, 8),
        kernel_size=3,
    )
    model_params['model_name'] = 'tf_efficientnet_b6_ns'


class MultiInstance04(Nspec16):
    """
    多实例学习配置 - MultiInstance04
    
    主要特点：
    - 使用MultiInstanceSCNN模型（将频谱图分成多个patch）
    - 使用XCiT（Cross-Covariance Image Transformer）作为backbone
    - 每个样本提取2个patch（n_patch=2），增强模型对局部特征的关注
    - 适合处理长时程信号
    """
    name = 'multi_instance_04'
    model = MultiInstanceSCNN  # 多实例CNN模型
    model_params = dict(
        model_name='xcit_tiny_12_p16_384_dist',  # XCiT Transformer模型
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(
            fs=2048, 
            lower_freq=16, 
            upper_freq=1024, 
            wavelet_width=8,
            stride=4,
            n_scales=384  # 384个尺度
        ),
        resize_img=(384, 768),  # 较大的图像尺寸
        n_patch=2,  # 将图像分成2个patch进行多实例学习
        custom_classifier='gem', 
        upsample='bicubic',
    )
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


'''
Sequential model - 1D序列模型配置
这些配置直接处理时域信号，不使用时频变换
'''
class Seq00(Resized08aug4):
    """
    1D ResNet配置 - Seq00
    
    主要特点：
    - 直接处理时域信号（3个探测器通道）
    - 使用1D ResNet架构，无需时频变换
    - 更快的训练速度（无需计算频谱图）
    """
    name = 'seq_00'
    model = ResNet1d  # 1D ResNet模型
    model_params = dict(
        in_channels=3,      # 3个探测器通道（LIGO Hanford, LIGO Livingston, Virgo）
        base_filters=64,    # 基础滤波器数量
        kernel_size=16,     # 卷积核大小
        stride=2,           # 步长
        groups=64,          # 分组卷积的组数
        n_block=16,         # ResNet块的数量
        n_classes=1,        # 输出类别数
        use_bn=True,        # 使用批归一化
        dropout=0.2         # Dropout比率
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=350, order=4),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=350, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=350, order=4),
        ])
    )
    dataset_params = dict()
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)
    num_epochs = 5
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)


class Seq02(Seq00):
    """
    1D ResNet的轻量级变体 - Seq02
    
    主要变化：
    - 减少基础滤波器到32（从64）
    - 减少分组卷积组数到32（从64）
    - 移除Dropout（dropout=0.0）
    - 调整带通滤波器范围到24-300 Hz
    - 适合需要更快训练速度的场景
    """
    name = 'seq_02'
    model_params = Seq00.model_params.copy()
    model_params.update(dict(
        base_filters=32,
        groups=32, 
        dropout=0.0
    ))
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=300, order=4),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=300, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=300, order=4),
        ])
    )


class Seq03(Seq02):
    """
    1D ResNet的高容量变体 - Seq03
    
    主要变化：
    - 增加基础滤波器到128（从32）
    - 调整带通滤波器范围到30-300 Hz
    - 添加高斯噪声增强（50%概率）
    - 增加训练轮数到8
    - 使用梯度裁剪（max_grad_norm=1000）稳定训练
    - 适合需要更强模型容量的场景
    """
    name = 'seq_03'
    model_params = Seq02.model_params.copy()
    model_params.update(dict(
        base_filters=128,
        groups=32,
        dropout=0.0
    ))
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ])
    )
    num_epochs = 8
    scheduler_params = dict(T_0=8, T_mult=1, eta_min=1e-6)
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)
    max_grad_norm = 1000
    clip_grad = 'value'


class Seq03aug3(Seq03):
    """
    Seq03的数据增强变体 - 使用波形翻转
    
    主要变化：
    - 用FlipWave替代GaussianNoiseSNR增强
    - FlipWave通过翻转波形增加数据多样性
    - 减少训练轮数到5（数据增强可能加速收敛）
    - 适合需要不同增强策略的场景
    """
    name = 'seq_03_aug_3'
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
            FlipWave(p=0.5)
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ])
    )
    num_epochs = 5
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)


class Seq09(Seq02):
    """
    1D DenseNet配置 - Seq09
    
    主要特点：
    - 使用DenseNet121-1D架构（密集连接的1D网络）
    - DenseNet通过特征重用提升参数效率
    - 直接处理时域信号，无需时频变换
    - 使用默认的DenseNet121-1D参数
    """
    name = 'seq_09'
    model = densenet121_1d
    model_params = dict()
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ])
    )


class Seq12(Seq02):
    """
    1D WaveNet配置 - Seq12
    
    主要特点：
    - 使用WaveNet1d架构（专为序列数据设计）
    - 简化的WaveNet块（simplified）
    - 逐渐增加的隐藏维度（128->256->512->1024）
    """
    name = 'seq_12'
    model = WaveNet1d  # 1D WaveNet模型
    model_params = dict(
        in_channels=3, 
        hidden_dims=(128, 256, 512, 1024),  # 各层的隐藏维度
        wave_block='simplified',            # 简化的WaveNet块
        num_classes=1,
        reinit=True,    # 重新初始化权重
        downsample=True, # 使用下采样
    )
    num_epochs = 5
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    max_grad_norm = 1000
    clip_grad = 'value'


class Seq12arch4(Seq12):
    """
    Seq12的深度变体 - 更深的WaveNet架构
    
    主要变化：
    - 增加网络深度：8层（从4层）
    - 更细粒度的隐藏维度变化（128->128->256->256->512->512->1024->1024）
    - 每层使用不同数量的WaveNet块（12, 10, 8, 8, 6, 6, 4, 2）
    - 设置kernel_size=16
    - 适合需要极深网络的复杂模式识别场景
    """
    name = 'seq_12_arch_4'
    model_params = Seq12.model_params.copy()
    model_params.update(dict(
        kernel_size=16,
        hidden_dims=(128, 128, 256, 256, 512, 512, 1024, 1024),
        wave_layers=(12, 10, 8, 8, 6, 6, 4, 2)
    ))


'''
PSEUDO LABEL - 伪标签训练配置
这些配置使用已训练模型的预测结果作为伪标签，在测试集上进行半监督学习
'''
class Pseudo06(Nspec12):
    """
    伪标签训练配置 - Pseudo06
    
    主要特点：
    - 基于Nspec12模型的预测结果生成伪标签
    - 使用软标签（hard_label=False），保留预测概率
    - 可以增加训练数据量，提升模型泛化能力
    """
    name = 'pseudo_06'
    weight_path = None  # 不使用预训练权重（从头训练）
    pseudo_labels = dict(
        path=Path('results/nspec_12/predictions.npy'),  # 伪标签文件路径
        confident_samples=None,  # 置信度阈值（None表示使用所有样本）
        hard_label=False  # False表示软标签（保留概率），True表示硬标签（0/1）
    )


class Pseudo07(Nspec16):
    """
    伪标签训练配置 - 基于Nspec16
    
    主要特点：
    - 使用Nspec16模型的预测结果作为伪标签
    - 从头训练（weight_path=None）
    - 使用软标签（hard_label=False）
    """
    name = 'pseudo_07'
    weight_path = None
    pseudo_labels = dict(
        path=Path('results/nspec_16/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )


class Pseudo10(Nspec16spec13):
    """
    伪标签训练配置 - 基于Nspec16spec13
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）提升泛化能力
    - 增加训练轮数到10以充分利用伪标签数据
    """
    name = 'pseudo_10'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_16_spec_13/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo12(Nspec12arch0):
    """
    伪标签训练配置 - 基于Nspec12arch0（DenseNet201）
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_12'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_12_arch_0/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo13(MultiInstance04):
    """
    伪标签训练配置 - 基于MultiInstance04（多实例学习）
    
    主要特点：
    - 使用多实例学习模型的预测结果作为伪标签
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_13'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/multi_instance_04/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo14(Nspec16arch17):
    """
    伪标签训练配置 - 基于Nspec16arch17（EfficientNet-B7-NS）
    
    主要特点：
    - 使用大模型的预测结果作为伪标签
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_14'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_16_arch_17/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo15(Nspec22aug1):
    """
    伪标签训练配置 - 基于Nspec22aug1（WaveNet + 翻转增强）
    
    主要特点：
    - 使用WaveNet时频变换模型的预测结果
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_15'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_aug_1/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo16(Nspec22arch2):
    """
    伪标签训练配置 - 基于Nspec22arch2（EfficientNet-B6-NS）
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_16'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_2/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo17(Nspec23arch3):
    """
    伪标签训练配置 - 基于Nspec23arch3（CNN频谱图 + EfficientNet-B6-NS）
    
    主要特点：
    - 使用CNN频谱图模型的预测结果
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_17'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_23_arch_3/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo18(Nspec21):
    """
    伪标签训练配置 - 基于Nspec21（EfficientNet-B4-NS + CWT）
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_18'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_21/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo19(Nspec22arch6):
    """
    伪标签训练配置 - 基于Nspec22arch6（DenseNet201 + WaveNet）
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_19'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_6/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo21(Nspec22arch7):
    """
    伪标签训练配置 - 基于Nspec22arch7（EfficientNetV2-M + WaveNet）
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_21'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_7/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo22(Nspec23arch5):
    """
    伪标签训练配置 - 基于Nspec23arch5（EfficientNetV2-M + CNN频谱图）
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_22'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_23_arch_5/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo23(Nspec22arch12):
    """
    伪标签训练配置 - 基于Nspec22arch12（EfficientNetV2-L + WaveNet）
    
    主要特点：
    - 使用大模型的预测结果作为伪标签
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_23'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_12/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo24(Nspec30arch2):
    """
    伪标签训练配置 - 基于Nspec30arch2（分离通道WaveNet）
    
    主要特点：
    - 使用分离通道WaveNet模型的预测结果
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_24'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_30_arch_2/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo25(Nspec25arch1):
    """
    伪标签训练配置 - 基于Nspec25arch1（改进的WaveNet频谱图）
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_25'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_25_arch_1/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo26(Nspec22arch10):
    """
    伪标签训练配置 - 基于Nspec22arch10（ResNet200D + WaveNet）
    
    主要特点：
    - 使用深度ResNet模型的预测结果
    - 使用平滑标签损失（smooth_eps=0.025）
    - 增加训练轮数到10
    """
    name = 'pseudo_26'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_10/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class PseudoSeq03(Seq09):
    """
    伪标签训练配置 - 基于Seq09（1D DenseNet）
    
    主要特点：
    - 使用1D序列模型的预测结果作为伪标签
    - 使用平滑标签损失（smooth_eps=0.025）
    """
    name = 'pseudo_seq_03'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/seq_09/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )


class PseudoSeq04(Seq03aug3):
    """
    伪标签训练配置 - 基于Seq03aug3（1D ResNet + 翻转增强）
    
    主要特点：
    - 使用平滑标签损失（smooth_eps=0.025）
    """
    name = 'pseudo_seq_04'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/seq_03_aug_3/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )


class PseudoSeq07(Seq12arch4):
    """
    伪标签训练配置 - 基于Seq12arch4（深度1D WaveNet）
    
    主要特点：
    - 使用深度WaveNet模型的预测结果
    - 使用平滑标签损失（smooth_eps=0.025）
    - 训练6个epoch（相比其他伪标签配置更少）
    """
    name = 'pseudo_seq_07'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/seq_12_arch_4/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 6
    scheduler_params = dict(T_0=6, T_mult=1, eta_min=1e-6)


class Debug(Seq12):
    """
    调试配置 - Debug
    
    用于快速测试代码和配置是否正确
    - 只训练2个epoch
    - 启用debug模式（可能包含额外的调试信息）
    """
    name = 'debug'
    debug = True  # 启用调试模式
    num_epochs = 2  # 只训练2个epoch用于快速测试
