# @Date 2022/7/30
# @Author Meimin Wang
# @Filename image_folder_dataset.py
# @Software PyCharm
# @Email blessed@gmail.com
# @Email wangmeimin@cstor.cn
from __future__ import annotations

import numpy as np
import tensorflow as tf
import os.path
from os import PathLike
from pathlib import Path


class ImageFolderDataset:
    def __init__(self,
                 root_dir: str | Path | PathLike
                 ) -> None:
        if not isinstance(root_dir, (str, Path, PathLike)):
            raise TypeError(f'The data root directory must be type of str, pathlib.Path or os.PathLike: {type(root_dir)}')
        if isinstance(root_dir, Path):
            root_dir = str(root_dir)
        elif isinstance(root_dir, PathLike):
            root_dir = root_dir.__fspath__()
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f'The data root directory is not found: {root_dir}')
        self.data_root = root_dir

    def load(self):
        tf.data.Dataset.list_files(
            os.path.join(self.data_root, '*', '*')
        )
