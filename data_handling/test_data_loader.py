from torch.utils.data.dataloader import DataLoader
from functools import partial
from .clotho_test_set import ClothoTestset
from .collate_fn_test import clotho_collate_fn_test


def get_test_data_loader(data_dir, batch_size, nb_t_steps_pad, shuffle, drop_last, input_pad_at='start', num_workers=0):
    dataset = ClothoTestset(data_dir)

    collate_fn = partial(
        clotho_collate_fn_test,
        nb_t_steps=nb_t_steps_pad,
        input_pad_at=input_pad_at)

    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        drop_last=drop_last, collate_fn=collate_fn)