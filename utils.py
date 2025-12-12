from pprint import pformat
import types
import os
import requests


def print_config(cfg, logger=None):
    """打印配置对象中的关键字段，支持写入外部 logger。"""

    def _print(text):
        if logger is None:
            print(text)
        else:
            logger(text)
    
    items = [
        'name', 
        'cv', 'num_epochs', 'batch_size', 'seed',
        'dataset', 'dataset_params', 'num_classes', 'transforms', 'splitter',
        'model', 'model_params', 'weight_path', 'optimizer', 'optimizer_params',
        'scheduler', 'scheduler_params', 'batch_scheduler', 'scheduler_target',
        'criterion', 'eval_metric', 'monitor_metrics',
        'amp', 'parallel', 'hook', 'callbacks', 'deterministic', 
        'clip_grad', 'max_grad_norm',
        'pseudo_labels'
    ]
    _print('===== CONFIG =====')
    for key in items:
        try:
            val = getattr(cfg, key)
            if isinstance(val, (type, types.FunctionType)):
                val = val.__name__ + '(*)'
            if isinstance(val, (dict, list)):
                val = '\n'+pformat(val, compact=True, indent=2)
            _print(f'{key}: {val}')
        except:
            _print(f'{key}: ERROR')
    _print(f'===== CONFIGEND =====')


def notify_me(text):
    """通知占位函数：可在此接入企业微信/飞书/钉钉/Line 等推送。"""
    # 示例：LINE Notify
    # line_notify_token = '{Your token}'  # 放置你的 token
    # line_notify_api = 'https://notify-api.line.me/api/notify'
    # headers = {'Authorization': f'Bearer {line_notify_token}'}
    # data = {'message': '\n' + text}     # 在消息前加换行方便阅读
    # requests.post(line_notify_api, headers=headers, data=data)
    pass
