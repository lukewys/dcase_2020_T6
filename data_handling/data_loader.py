from typing import Tuple, List, AnyStr, Union
from pathlib import Path

from numpy import ndarray, recarray
from torch.utils.data import Dataset
from numpy import load as np_load

import torch
import numpy as np
import os
from typing import MutableSequence, Union, Tuple, AnyStr,Optional,Callable
from numpy import ndarray

from torch import cat as pt_cat, zeros as pt_zeros, \
    ones as pt_ones, from_numpy, Tensor

from functools import partial
from pathlib import Path

from torch.utils.data.dataloader import DataLoader


def get_all_ref(filename, data_dir):
    filename = str(filename)
    tgt = [np.load(d, allow_pickle=True)[0][4].tolist()
           for d in [os.path.join(data_dir, 'clotho_file_{filename}.wav_{i}.npy'.
                                  format(filename=filename[:-4],  # 删除'.wav'
                                         i=i)) for i in range(5)]  # wav_0-wav_4
           ]
    return tgt

def get_tag(captions):
    pass

#input_field_name='features',
#output_field_name='words_ind',

class ClothoDataset(Dataset):

    def __init__(self, data_dir: Path,
                 split: AnyStr,
                 input_field_name: AnyStr,
                 output_field_name: AnyStr,
                 load_into_memory: bool) \
            -> None:
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
        return len(self.examples)

    def __getitem__(self,item: int):
        ex: Union[Path, recarray] = self.examples[item]
        if not self.load_into_memory:
            ex: recarray = np_load(str(ex), allow_pickle=True)

        in_e, ou_e = [ex[i].item() for i in [self.input_name, self.output_name]]
        out_len = len(ou_e)
        all_ref = get_all_ref(ex['file_name'].item(), self.data_dir)

        filename = str(ex['file_name'].item())
        return in_e, ou_e, out_len



def clotho_collate_fn(batch: MutableSequence[ndarray],
                      nb_t_steps: Union[AnyStr, Tuple[int, int]],
                      input_pad_at: str,
                      output_pad_at: str):
    """Pads data.

    :param batch: Batch data.
    :type batch: list[numpy.ndarray]
    :param nb_t_steps: Number of time steps to\
                       pad/truncate to. Cab use\
                       'max', 'min', or exact number\
                       e.g. (1024, 10).
    :type nb_t_steps: str|(int, int)
    :param input_pad_at: Pad input at the start or\
                         at the end?
    :type input_pad_at: str
    :param output_pad_at: Pad output at the start or\
                          at the end?
    :type output_pad_at: str
    :return: Padded data.
    :rtype: torch.Tensor, torch.Tensor
    """
    if type(nb_t_steps) == str:
        truncate_fn = max if nb_t_steps.lower() == 'max' else min
        in_t_steps = truncate_fn([i[0].shape[0] for i in batch])
        out_t_steps = truncate_fn([i[1].shape[0] for i in batch])
    else:
        in_t_steps, out_t_steps = nb_t_steps

    in_dim = batch[0][0].shape[-1]
    eos_token = batch[0][1][-1]

    batch = sorted(batch, key=lambda x: x[-1])
    input_tensor, output_tensor = [], []

    for in_b, out_b, out_len in batch:
        if in_t_steps >= in_b.shape[0]:
            padding = pt_zeros(in_t_steps - in_b.shape[0], in_dim).float()
            data = [from_numpy(in_b).float()]
            if input_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)
            tmp_in: Tensor = pt_cat(data)
        else:
            tmp_in: Tensor = from_numpy(in_b[:in_t_steps, :]).float()
        input_tensor.append(tmp_in.unsqueeze_(0))

        if out_t_steps >= out_b.shape[0]:
            padding = pt_ones(out_t_steps - len(out_b)).mul(eos_token).long()
            data = [from_numpy(out_b).long()]
            if output_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)

            tmp_out: Tensor = pt_cat(data)
        else:
            tmp_out: Tensor = from_numpy(out_b[:out_t_steps]).long()
        output_tensor.append(tmp_out.unsqueeze_(0))


    input_tensor = pt_cat(input_tensor)
    output_tensor = pt_cat(output_tensor)
    *_, output_len = zip(*batch)
    output_len = torch.LongTensor(output_len)
    return input_tensor, output_tensor, output_len
