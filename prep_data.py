"""
数据准备脚本 - prep_data.py

本脚本的主要功能：
1. 读取 Kaggle 竞赛的原始数据文件（training_labels.csv 和 sample_submission.csv）
2. 为每个样本生成对应的波形文件路径（根据 ID 的前三个字符组织目录结构）
3. 生成新的 train.csv 和 test.csv 文件，包含 id、target（仅训练集）和 path 列
4. 可选：生成波形缓存文件（pickle 格式），用于加速训练时的数据加载

使用示例：
    python prep_data.py                          # 仅生成 CSV 文件
    python prep_data.py --cache                  # 同时生成缓存文件
    python prep_data.py --hardware RTX3090       # 指定硬件配置（影响缓存大小）
"""

import argparse
from pathlib import Path
import pandas as pd
import pickle
import gc

from kuma_utils.torch import TorchLogger

from configs import HW_CFG  # 硬件配置字典，定义不同硬件的 CPU/内存/GPU 规格
from datasets import load_signal_cache  # 加载波形数据到内存缓存的函数


if __name__ == "__main__":
    # ========== 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description='准备 G2Net 竞赛数据文件')
    parser.add_argument("--root_dir", type=str, default='/kaggle/working/kaggle-g2net-public/input/',
                        help="原始数据目录（包含 training_labels.csv 和 sample_submission.csv）")
    parser.add_argument("--export_dir", type=str, default='/kaggle/working/kaggle-g2net-public/input/',
                        help="输出目录（生成的 train.csv、test.csv 和缓存文件将保存在这里）")
    parser.add_argument("--hardware", type=str, default='A100',
                        help="硬件配置名称（从 configs.py 的 HW_CFG 中选择，决定缓存大小限制）")
    parser.add_argument("--cache", action='store_true', 
                        help="是否生成波形缓存文件（需要至少 32GB RAM，可显著加速训练）")
   
    opt = parser.parse_args()
    LOGGER = TorchLogger('tmp.log', file=False)  # 初始化日志记录器（不写入文件）

    # ========== 获取硬件配置 ==========
    # HW_CFG 格式：(CPU核心数, RAM大小(GB), GPU数量, GPU显存总量(GB))
    N_CPU, N_RAM, N_GPU, N_GRAM = HW_CFG[opt.hardware] # N_GPU, N_GRAM好像并没有用
    if opt.cache:
        # 缓存大小限制为 RAM 的一半，避免内存溢出
        LOGGER(f'最大缓存大小设置为 {N_RAM//2} GB')

    # ========== 设置目录路径 ==========
    root_dir = Path(opt.root_dir).expanduser()  # 展开用户目录路径（如 ~/data -> /home/user/data）
    
    # ========== 读取原始数据文件 ==========
    train = pd.read_csv(root_dir/'training_labels.csv')  # 训练集标签（包含 id 和 target 列）
    test = pd.read_csv(root_dir/'sample_submission.csv')  # 测试集提交模板（包含 id 和 target 列）
    
    export_dir = Path(opt.export_dir).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录（如果不存在）

    # ========== 处理训练集 ==========
    LOGGER('===== TRAIN =====')
    # 根据 ID 的前三个字符创建三级目录结构
    # 例如：ID = "abc123def" -> train/a/b/c/abc123def.npy
    # 这种组织方式可以避免单个目录下文件过多，提高文件系统性能
    train['path'] = train['id'].apply(
        lambda x: root_dir/f'train/{x[0]}/{x[1]}/{x[2]}/{x}.npy'
    )
    # 保存包含路径信息的训练集 CSV 文件
    train.to_csv(export_dir/'train.csv', index=False)
    
    # ========== 可选：生成训练集波形缓存 ==========
    if opt.cache:
        # load_signal_cache 函数会：
        # 1. 并行加载所有波形文件（.npy 格式）到内存
        # 2. 限制总缓存大小不超过 N_RAM//2 GB
        # 3. 返回一个字典：{文件路径: 波形数据数组}
        train_cache = load_signal_cache(
            train['path'].values,  # 所有训练集文件路径
            N_RAM//2,              # 缓存大小限制（GB）
            n_jobs=N_CPU           # 并行加载的进程数
        )
        # 将缓存保存为 pickle 文件，训练时可以直接加载到内存
        with open(export_dir/'train_cache.pickle', 'wb') as f:
            pickle.dump(train_cache, f)
        # 释放内存
        del train_cache
        gc.collect()

    # ========== 处理测试集 ==========
    LOGGER('===== TEST =====')
    # 同样根据 ID 的前三个字符创建目录结构
    test['path'] = test['id'].apply(
        lambda x: root_dir/f'test/{x[0]}/{x[1]}/{x[2]}/{x}.npy'
    )
    # 保存包含路径信息的测试集 CSV 文件
    test.to_csv(export_dir/'test.csv', index=False)
    
    # ========== 可选：生成测试集波形缓存 ==========
    if opt.cache:
        test_cache = load_signal_cache(
            test['path'].values,   # 所有测试集文件路径
            N_RAM//2,              # 缓存大小限制（GB）
            n_jobs=N_CPU           # 并行加载的进程数
        )
        # 将缓存保存为 pickle 文件
        with open(export_dir/'test_cache.pickle', 'wb') as f:
            pickle.dump(test_cache, f)
        # 释放内存
        del test_cache
        gc.collect()
    