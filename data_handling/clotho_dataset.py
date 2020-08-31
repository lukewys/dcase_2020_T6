#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, List, AnyStr, Union
from pathlib import Path

from numpy import ndarray, recarray
from torch.utils.data import Dataset
from numpy import load as np_load

import torch
import numpy as np
import os

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ClothoDataset']


class ClothoDataset(Dataset):

    def __init__(self, data_dir: Path,
                 split: AnyStr,
                 input_field_name: AnyStr,
                 output_field_name: AnyStr,
                 load_into_memory: bool) \
            -> None:
        """Initialization of a Clotho dataset object.

        :param data_dir: Directory with data.
        :type data_dir: pathlib.Path
        :param split: Split to use (i.e. 'development', 'evaluation')
        :type split: str
        :param input_field_name: Field name of the clotho data\
                                 to be used as input data to the\
                                 method.
        :type input_field_name: str
        :param output_field_name: Field name of the clotho data\
                                 to be used as output data to the\
                                 method.
        :type output_field_name: str
        :param load_into_memory: Load all data into memory?
        :type load_into_memory: bool
        """
        super(ClothoDataset, self).__init__()
        the_dir: Path = data_dir.joinpath(split)

        self.examples: List[Path] = sorted(the_dir.iterdir())
        self.input_name: str = input_field_name
        self.output_name: str = output_field_name
        self.load_into_memory: bool = load_into_memory

        if load_into_memory:
            self.examples: List[recarray] = [np_load(str(f), allow_pickle=True)
                                             for f in self.examples]

    def __len__(self) \
            -> int:
        """Gets the amount of examples in the dataset.

        :return: Amount of examples in the dataset.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self,
                    item: int) \
            -> Tuple[ndarray, ndarray]:
        """Gets an example from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Input and output values.
        :rtype: numpy.ndarray. numpy.ndarray
        """
        ex: Union[Path, recarray] = self.examples[item]
        if not self.load_into_memory:
            ex: recarray = np_load(str(ex), allow_pickle=True)

        in_e, ou_e = [ex[i].item() for i in [self.input_name, self.output_name]]

        return in_e, ou_e


class ClothoDatasetEval(Dataset):

    def __init__(self, data_dir: Path,
                 split: AnyStr,
                 input_field_name: AnyStr,
                 output_field_name: AnyStr,
                 load_into_memory: bool) \
            -> None:
        """Initialization of a Clotho dataset object.

        :param data_dir: Directory with data.
        :type data_dir: pathlib.Path
        :param split: Split to use (i.e. 'development', 'evaluation')
        :type split: str
        :param input_field_name: Field name of the clotho data\
                                 to be used as input data to the\
                                 method.
        :type input_field_name: str
        :param output_field_name: Field name of the clotho data\
                                 to be used as output data to the\
                                 method.
        :type output_field_name: str
        :param load_into_memory: Load all data into memory?
        :type load_into_memory: bool
        """
        super(ClothoDatasetEval, self).__init__()
        the_dir: Path = data_dir.joinpath(split)
        if split == 'evaluation':
            self.examples: List[Path] = sorted(the_dir.iterdir())[::5]  # changed
        else:
            self.examples: List[Path] = sorted(the_dir.iterdir())  # changed
        # self.examples: List[Path] = sorted(the_dir.iterdir())
        self.input_name: str = input_field_name
        self.output_name: str = output_field_name
        self.load_into_memory: bool = load_into_memory
        self.data_dir = the_dir

        if load_into_memory:
            self.examples: List[recarray] = [np_load(str(f), allow_pickle=True)
                                             for f in self.examples]

    def __len__(self) \
            -> int:
        """Gets the amount of examples in the dataset.

        :return: Amount of examples in the dataset.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self,
                    item: int):
        """Gets an example from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Input and output values.
        :rtype: numpy.ndarray. numpy.ndarray
        """
        ex: Union[Path, recarray] = self.examples[item]
        if not self.load_into_memory:
            ex: recarray = np_load(str(ex), allow_pickle=True)

        in_e, ou_e = [ex[i].item() for i in [self.input_name, self.output_name]]

        all_ref = get_all_ref(ex['file_name'].item(), self.data_dir)

        filename = str(ex['file_name'].item())
        out_len = len(ou_e)
        return in_e, ou_e, all_ref, filename,out_len


def get_all_ref(filename, data_dir):
    filename = str(filename)
    # tgt = [np.load(d, allow_pickle=True).words_ind.tolist()
    tgt = [np.load(d, allow_pickle=True)['words_ind'].item().tolist()
           for d in [os.path.join(data_dir, 'clotho_file_{filename}.wav_{i}.npy'.
                                  format(filename=filename[:-4],  # 删除'.wav'
                                         i=i)) for i in range(5)]  # wav_0-wav_4
           ]
    return tgt
# EOF
