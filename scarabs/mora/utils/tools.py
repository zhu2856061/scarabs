# -*- coding: utf-8 -*-
# @Time   : 2024/08/26 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os


# 获得文件夹下，指定后缀的文件路径
def get_filenames(directory, suffix=None):
    filenames = []
    files = os.listdir(directory)
    for _file in files:
        tmp_file = os.path.join(directory, _file)
        if os.path.isfile(tmp_file):
            if tmp_file.endswith(suffix):
                filenames.append(tmp_file)
    return filenames


# 获得文件夹下，最大的checkpoint的文件
def get_checkpoint_path(directory):
    filenames = []
    files = os.listdir(directory)
    for _file in files:
        if _file.startswith("checkpoint"):
            filenames.append(_file)

    return os.path.join(directory, max(filenames))


# 计算模型参数量的方法
def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"层: {name} | 参数大小: {param.size()} | 参数量: {param.numel()}")

    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {count}")


# 设置日志颜色
def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except Exception:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"
