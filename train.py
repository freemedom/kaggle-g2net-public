import argparse
from pathlib import Path
from pprint import pprint
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import pickle
import traceback

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.utils import get_time, seed_everything, fit_state_dict

from configs import *
from transforms import Compose, FlipWave
from utils import print_config, notify_me
from training_extras import make_tta_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 解析命令行参数以选择配置、硬件、推理/训练模式等
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--hardware", type=str, default='A100',
                        help="hardware name (this determines num of cpus and gpus)")
    parser.add_argument("--limit_fold", type=int, default=-1,
                        help="train only specified fold")
    parser.add_argument("--inference", action='store_true',
                        help="inference")
    parser.add_argument("--tta", action='store_true', 
                        help="test time augmentation ")
    parser.add_argument("--gpu", nargs="+", default=[])
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--wait", type=int, default=0,
                        help="time (sec) to wait before execution")
    opt = parser.parse_args()
    pprint(opt)

    ''' Configure hardware '''
    N_CPU, N_RAM, N_GPU, N_GRAM = HW_CFG[opt.hardware]  # 根据硬件预设，确定可用资源
    if len(opt.gpu) == 0:
        opt.gpu = None # use all GPUs
    elif len(opt.gpu) > N_GPU:
        raise ValueError(f'Maximum GPUs allowed is {N_GPU}')

    ''' Configure path '''
    cfg = eval(opt.config)  # 动态获取 configs.py 中的配置类实例
    assert cfg.pseudo_labels is None
    export_dir = Path('results') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)  # 结果目录

    ''' Configure logger '''
    log_items = [
        'epoch', 'train_loss', 'train_metric', 'train_monitor', 
        'valid_loss', 'valid_metric', 'valid_monitor', 
        'learning_rate', 'early_stop'
    ]
    if opt.debug:
        log_items += ['gpu_memory']
    if opt.limit_fold >= 0:
        logger_path = f'{cfg.name}_fold{opt.limit_fold}_{get_time("%y%m%d%H%M")}.log'
    else:
        logger_path = f'{cfg.name}_{get_time("%y%m%d%H%M")}.log'
    LOGGER = TorchLogger(
        export_dir / logger_path, 
        log_items=log_items, file=not opt.silent
    )
    if opt.wait > 0:
        LOGGER(f'Waiting for {opt.wait} sec.')
        time.sleep(opt.wait)

    ''' Prepare data '''
    seed_everything(cfg.seed, cfg.deterministic)
    print_config(cfg, LOGGER)
    train = pd.read_csv(cfg.train_path)  # 训练集 CSV（包含 path, target）
    test = pd.read_csv(cfg.test_path)    # 测试集 CSV（只含 path）
    if cfg.debug:
        train = train.iloc[:10000]
        test = test.iloc[:1000]
    splitter = cfg.splitter
    fold_iter = list(splitter.split(X=train, y=train['target']))  # 预先生成全部折分
    
    '''
    Training
    '''
    scores = []
    if cfg.train_cache is None:
        train_cache = None
    else:
        with open(cfg.train_cache, 'rb') as f:
            train_cache = pickle.load(f)
    if cfg.test_cache is None:
        test_cache = None
    else:
        with open(cfg.test_cache, 'rb') as f:
            test_cache = pickle.load(f)
    
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):
        # 中文：支持 limit_fold 只跑指定折；inference 模式直接跳过训练；skip_existing 避免覆盖已有模型
        
        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            continue  # skip fold

        if opt.inference:
            continue

        if opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'checkpoint fold{fold}.pt already exists.')
            continue

        LOGGER(f'===== TRAINING FOLD {fold} =====')

        train_fold = train.iloc[train_idx]
        valid_fold = train.iloc[valid_idx]

        # 中文：打印当前折的正样本比例与样本量，便于检查折分是否均衡
        LOGGER(f'train positive: {train_fold.target.values.mean(0)} ({len(train_fold)})')
        LOGGER(f'valid positive: {valid_fold.target.values.mean(0)} ({len(valid_fold)})')

        train_data = cfg.dataset(
            paths=train_fold['path'].values, targets=train_fold['target'].values,
            transforms=cfg.transforms['train'], cache=train_cache, is_test=False,
            **cfg.dataset_params)
        valid_data = cfg.dataset(
            paths=valid_fold['path'].values, targets=valid_fold['target'].values,
            transforms=cfg.transforms['test'], cache=train_cache, is_test=True,
            **cfg.dataset_params)

        # 中文：DataLoader 配置说明
        # - num_workers=0：主进程加载数据，避免多进程问题（Windows/调试场景）
        # - pin_memory=False：不使用固定内存（pinned memory）
        #   * pin_memory=True 时，数据固定在内存中，CPU→GPU 传输更快，但占用更多内存
        #   * pin_memory=False 时，使用常规内存，传输较慢但更省内存，适合调试或内存受限场景
        '''
        普通内存（页式内存）：
        操作系统可能将数据从物理内存换到磁盘（虚拟内存/swap）
        GPU 无法直接访问虚拟内存，需要先换回物理内存，传输较慢
        固定内存（pinned memory）：
        告诉操作系统“这段内存不要换出”，始终保持在物理内存中
        GPU 可通过 DMA（直接内存访问）直接从物理内存读取，传输更快
        '''
        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=False)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)

        model = cfg.model(**cfg.model_params)  # 构建模型

        # Load snapshot
        # 中文：可选加载预训练/历史权重，若传入目录则按折号加载对应权重
        if cfg.weight_path is not None:
            if cfg.weight_path.is_dir():
                weight_path = cfg.weight_path / f'fold{fold}.pt'
            else:
                weight_path = cfg.weight_path
            LOGGER(f'{weight_path} loaded.')
            weight = torch.load(weight_path, 'cpu')['model']
            fit_state_dict(weight, model)
            model.load_state_dict(weight, strict=False)
            del weight; gc.collect()
        # Load SeqCNN model
        # 中文：若模型包含子模块路径，则分别加载 CNN/Seq 子模型；部分模型支持 freeze 冻结
        if hasattr(model, 'cnn_path'):
            checkpoint = torch.load(model.cnn_path / f'fold{fold}.pt')['model']
            model.load_cnn(checkpoint)
        if hasattr(model, 'seq_path'):
            checkpoint = torch.load(model.seq_path / f'fold{fold}.pt')['model']
            model.load_seq(checkpoint)
        if hasattr(model, 'freeze'):
            model.freeze_seq()
            model.freeze_cnn()

        optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)  # 优化器
        scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)          # 学习率调度器
        # 中文：封装训练所需参数，直接传递给 TorchTrainer
        FIT_PARAMS = {
            'loader': train_loader,
            'loader_valid': valid_loader,
            'criterion': cfg.criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scheduler_target': cfg.scheduler_target,
            'batch_scheduler': cfg.batch_scheduler, 
            'num_epochs': cfg.num_epochs,
            'callbacks': deepcopy(cfg.callbacks),
            'hook': cfg.hook,
            'export_dir': export_dir,
            'eval_metric': cfg.eval_metric,
            'monitor_metrics': cfg.monitor_metrics,
            'fp16': cfg.amp,
            'parallel': cfg.parallel,
            'deterministic': cfg.deterministic, 
            'clip_grad': cfg.clip_grad, 
            'max_grad_norm': cfg.max_grad_norm,
            'random_state': cfg.seed,
            'logger': LOGGER,
            'progress_bar': opt.progress_bar, 
            'resume': opt.resume
        }
        if not cfg.debug:
            notify_me(f'[{cfg.name}:fold{opt.limit_fold}]\nTraining started.')
        try:
            # 中文：实例化训练器（serial 区分折次、device 指定 GPU/CPU），开始单折训练
            trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
            trainer.fit(**FIT_PARAMS)  # 训练一个折
        except Exception as e:
            err = traceback.format_exc()
            LOGGER(err)
            if not opt.silent:
                notify_me('\n'.join([
                    f'[{cfg.name}:fold{opt.limit_fold}]', 
                    'Training stopped due to:', 
                    f'{traceback.format_exception_only(type(e), e)}'
                ]))
        # 中文：每折结束主动释放内存/显存，避免长时间训练累积占用
        del model, trainer, train_data, valid_data; gc.collect()
        torch.cuda.empty_cache()


    '''
    Inference
    '''
    # 预测缓存：
    # predictions[fold, sample, 0] 存每折对 test 的预测
    # outoffolds[sample, 0]        存 OOF 预测（与训练折分一致）
    predictions = np.full((cfg.cv, len(test), 1), 0.5, dtype=np.float32)
    outoffolds = np.full((len(train), 1), 0.5, dtype=np.float32)
    test_data = cfg.dataset(
        paths=test['path'].values, transforms=cfg.transforms['test'], 
        cache=test_cache, is_test=True, **cfg.dataset_params)
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):

        if opt.limit_fold >= 0:
            if fold == 0:
                checkpoint = torch.load(export_dir/f'fold{opt.limit_fold}.pt', 'cpu')
                scores.append(checkpoint['state']['best_score'])
            continue

        if not (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'fold{fold}.pt missing. No target to predict.')
            continue

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        valid_fold = train.iloc[valid_idx]
        valid_data = cfg.dataset(
            paths=valid_fold['path'].values, targets=valid_fold['target'].values,
            cache=train_cache, transforms=cfg.transforms['test'], is_test=True,
            **cfg.dataset_params)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)
        test_loader = D.DataLoader(
            test_data, batch_size=cfg.batch_size, shuffle=False, 
            num_workers=0, pin_memory=False)

        model = cfg.model(**cfg.model_params)  # 重建模型并加载权重
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        fit_state_dict(checkpoint['model'], model)
        try:
            model.load_state_dict(checkpoint['model'])
        except: # drop preprocess module for compatibility
            model.cnn = nn.Sequential(
                *[model.cnn[i+1] for i in range(len(model.cnn)-1)])
            model.load_state_dict(checkpoint['model'])
        scores.append(checkpoint['state']['best_score'])
        del checkpoint; gc.collect()

        trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
        trainer.register(hook=cfg.hook, callbacks=cfg.callbacks)  # 注册钩子与回调（早停/保存等）

        if opt.tta: # flip wave TTA：对 test/valid 各跑两次，取平均
            tta_transform = Compose(
                cfg.transforms['test'].transforms + [FlipWave(always_apply=True)])
            LOGGER(f'[{fold}] pred0 {test_loader.dataset.transforms}')
            prediction0 = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            test_loader = make_tta_dataloader(test_loader, cfg.dataset, dict(
                paths=test['path'].values, transforms=tta_transform, 
                cache=test_cache, is_test=True, **cfg.dataset_params
            ))
            LOGGER(f'[{fold}] pred1 {test_loader.dataset.transforms}')
            prediction1 = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            prediction_fold = (prediction0 + prediction1) / 2

            LOGGER(f'[{fold}] oof0 {valid_loader.dataset.transforms}')
            outoffold0 = trainer.predict(valid_loader, progress_bar=opt.progress_bar)
            valid_loader = make_tta_dataloader(valid_loader, cfg.dataset, dict(
                paths=valid_fold['path'].values, targets=valid_fold['target'].values,
                cache=train_cache, transforms=tta_transform, is_test=True,
                **cfg.dataset_params))
            LOGGER(f'[{fold}] oof1 {valid_loader.dataset.transforms}')
            outoffold1 = trainer.predict(valid_loader, progress_bar=opt.progress_bar)
            outoffold = (outoffold0 + outoffold1) / 2
        else:
            prediction_fold = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            outoffold = trainer.predict(valid_loader, progress_bar=opt.progress_bar)

        predictions[fold] = prediction_fold        # test 预测：按折存放，后续可对折/模型求平均
        outoffolds[valid_idx] = outoffold          # OOF (out-of-fold) 预测：对每折验证集的预测，写回原始索引；用于计算全体验证集的整体指标（如全数据 AUC）

        del model, trainer, valid_data; gc.collect()
        torch.cuda.empty_cache()

    if opt.limit_fold < 0:
        if opt.tta:
            np.save(export_dir/'outoffolds_tta', outoffolds)
            np.save(export_dir/'predictions_tta', predictions)
        else:
            np.save(export_dir/'outoffolds', outoffolds)
            np.save(export_dir/'predictions', predictions)

    LOGGER(f'scores: {scores}')  # 各折最佳分数
    LOGGER(f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}')
    if not cfg.debug:
        notify_me('\n'.join([
            f'[{cfg.name}:fold{opt.limit_fold}]',
            'Training has finished successfully.',
            f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}'
        ]))  # 训练完成后通知（如开启推送）
