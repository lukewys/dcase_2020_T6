#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain, count
from collections import deque, Counter
from pathlib import Path
from typing import MutableSequence, MutableMapping, \
    Tuple, List, Any

import numpy as np

from tools.csv_functions import read_csv_file
from tools.captions_functions import get_sentence_words, \
    clean_sentence, get_words_counter
from tools.file_io import load_numpy_object, load_audio_file, \
    load_pickle_file, dump_numpy_object, dump_pickle_file

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['check_data_for_split',
           'create_lists_and_frequencies',
           'create_split_data',
           'get_annotations_files',
           'get_amount_of_file_in_dir']


def check_data_for_split(dir_audio: Path,
                         dir_data: Path,
                         dir_root: Path,
                         csv_split: MutableSequence[MutableMapping[str, str]],
                         settings_ann: MutableMapping[str, Any],
                         settings_audio: MutableMapping[str, Any],
                         settings_cntr: MutableMapping[str, Any]) \
        -> None:
    """Goes through all audio files and checks the created data.

    Gets each audio file and checks if there are associated data.\
    If there are, checks the validity of the raw audio data and\
    the validity of the captions, words, and characters.

    :param dir_audio: Directory with the audio files.
    :type dir_audio: pathlib.Path
    :param dir_data: Directory with the data to be checked.
    :type dir_data: pathlib.Path
    :param dir_root: Root directory.
    :type dir_root: pathlib.Path
    :param csv_split: CSV entries for the data/
    :type csv_split: list[collections.OrderedDict]
    :param settings_ann: Settings for annotations.
    :type settings_ann: dict
    :param settings_audio: Settings for audio.
    :type settings_audio: dict
    :param settings_cntr: Settings for counters.
    :type settings_cntr: dict
    """
    # Load the words and characters lists
    words_list = load_pickle_file(dir_root.joinpath(
        settings_cntr['pickle_files_dir'],
        settings_cntr['files']['words_list_file_name']))
    chars_list = load_pickle_file(dir_root.joinpath(
        settings_cntr['pickle_files_dir'],
        settings_cntr['files']['characters_list_file_name']))

    data_files = list(dir_root.joinpath(dir_data).iterdir())

    for csv_entry in csv_split:
        # Get audio file name
        file_name_audio = Path(
            csv_entry[settings_ann['audio_file_column']])

        # Check if the audio file existed originally
        if not dir_audio.joinpath(file_name_audio).exists():
            raise FileExistsError(f'Audio file {file_name_audio} '
                                  f'not exists in {dir_audio}')

        # Flag for checking if there are data files for the audio file
        audio_has_data_files = False

        # Get the original audio data
        data_audio_original = load_audio_file(
            audio_file=str(dir_audio.joinpath(file_name_audio)),
            sr=int(settings_audio['sr']),
            mono=settings_audio['to_mono'])

        for data_file_index in range(len(data_files) - 1, -1, -1):
            # Get the stem of the audio file name
            f_stem = str(data_files[data_file_index]).split(
                'file_')[-1].split('.wav_')[0]

            if f_stem == file_name_audio.stem:
                audio_has_data_files = True

                data_file = data_files.pop(data_file_index)

                # Get the numpy record array
                data_array = load_numpy_object(data_file)

                # Get the audio data from the numpy record array
                data_audio_rec_array = data_array['audio_data'].item()

                # Compare the lengths
                if len(data_audio_rec_array) != len(data_audio_original):
                    raise ValueError(f'File {file_name_audio} was '
                                     f'not saved successfully to the '
                                     f'numpy object {data_file}.')

                # Check all elements, one to one
                if not all([data_audio_original[i] == data_audio_rec_array[i]
                            for i in range(len(data_audio_original))]):
                    raise ValueError(f'Numpy object {data_file} has '
                                     f'wrong audio data.')

                # Get the original caption
                caption_index = data_array['caption_ind'].item()

                # Clean it to remove any spaces before punctuation.
                original_caption = clean_sentence(
                    sentence=csv_entry[settings_ann[
                        'captions_fields_prefix'].format(
                        caption_index + 1)],
                    keep_case=True,
                    remove_punctuation=False,
                    remove_specials=not settings_ann[
                        'use_special_tokens'])

                # Check with the file caption
                caption_data_array = clean_sentence(
                    sentence=data_array['caption'].item(),
                    keep_case=True,
                    remove_punctuation=False,
                    remove_specials=not settings_ann[
                        'use_special_tokens'])

                if not original_caption == caption_data_array:
                    raise ValueError(f'Numpy object {data_file} '
                                     f'has wrong caption.')

                # Since caption in the file is OK, we can use it
                # instead of the original, because it already has
                # the special tokens.
                caption_data_array = clean_sentence(
                    sentence=data_array['caption'].item(),
                    keep_case=settings_ann['keep_case'],
                    remove_punctuation=settings_ann[
                        'remove_punctuation_words'],
                    remove_specials=not settings_ann[
                        'use_special_tokens'])

                # Check with the indices of words
                words_indices = data_array['words_ind'].item()
                caption_form_words = ' '.join([
                    words_list[i] for i in words_indices])

                if not caption_data_array == caption_form_words:
                    raise ValueError(f'Numpy object {data_file} '
                                     f'has wrong words indices.')

                # Check with the indices of characters
                caption_from_chars = ''.join([
                    chars_list[i] for i in data_array['chars_ind'].item()])

                caption_data_array = clean_sentence(
                    sentence=data_array['caption'].item(),
                    keep_case=settings_ann['keep_case'],
                    remove_punctuation=settings_ann[
                        'remove_punctuation_chars'],
                    remove_specials=not settings_ann[
                        'use_special_tokens'])

                if not caption_data_array == caption_from_chars:
                    raise ValueError(f'Numpy object {data_file} '
                                     f'has wrong characters indices.')

        if not audio_has_data_files:
            raise FileExistsError(f'Audio file {file_name_audio} has '
                                  f'no associated data.')


def create_lists_and_frequencies(captions: MutableSequence[str],
                                 dir_root: Path,
                                 settings_ann: MutableMapping[str, Any],
                                 settings_cntr: MutableMapping[str, Any]) -> \
        Tuple[List[str], List[str]]:
    """Creates the pickle files with words, characters, and their\
    frequencies.

    :param captions: Captions to be used (development captions are\
                     suggested).
    :type captions: list[str]
    :param dir_root: Root directory of data.
    :type dir_root: pathlib.Path
    :param settings_ann: Settings for annotations.
    :type settings_ann: dict
    :param settings_cntr: Settings for pickle files.
    :type settings_cntr: dict
    :return: Words and characters list.
    :rtype: list[str], list[str]
    """
    # Get words counter
    counter_words = get_words_counter(
        captions=captions,
        use_unique=settings_ann['use_unique_words_per_caption'],
        keep_case=settings_ann['keep_case'],
        remove_punctuation=settings_ann['remove_punctuation_words'],
        remove_specials=not settings_ann['use_special_tokens'])

    # Get words and frequencies
    words_list, frequencies_words = list(counter_words.keys()), \
                                    list(counter_words.values())

    # Get characters and frequencies
    cleaned_captions = [clean_sentence(
        sentence, keep_case=settings_ann['keep_case'],
        remove_punctuation=settings_ann['remove_punctuation_chars'],
        remove_specials=True)
        for sentence in captions]

    characters_all = list(chain.from_iterable(cleaned_captions))
    counter_characters = Counter(characters_all)

    # Add special characters
    if settings_ann['use_special_tokens']:
        counter_characters.update(['<sos>'] * len(cleaned_captions))
        counter_characters.update(['<eos>'] * len(cleaned_captions))

    chars_list, frequencies_chars = list(counter_characters.keys()), \
                                    list(counter_characters.values())

    # Save to disk
    obj_list = [words_list, frequencies_words, chars_list,
                frequencies_chars]
    obj_f_names = [
        settings_cntr['files']['words_list_file_name'],
        settings_cntr['files']['words_counter_file_name'],
        settings_cntr['files']['characters_list_file_name'],
        settings_cntr['files']['characters_frequencies_file_name']]

    output_dir = dir_root.joinpath(settings_cntr['pickle_files_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    [dump_pickle_file(obj=obj, file_name=output_dir.joinpath(obj_f_name))
     for obj, obj_f_name in zip(obj_list, obj_f_names)]

    return words_list, chars_list


def create_split_data(csv_split: MutableSequence[MutableMapping[str, str]],
                      dir_split: Path,
                      dir_audio: Path,
                      words_list: MutableSequence[str],
                      chars_list: MutableSequence[str],
                      settings_ann: MutableMapping[str, Any],
                      settings_audio: MutableMapping[str, Any],
                      settings_output: MutableMapping[str, Any]) \
        -> None:
    """Creates the data for the split.

    :param csv_split: Annotations of the split.
    :type csv_split: list[collections.OrderedDict]
    :param dir_split: Directory for the split.
    :type dir_split: pathlib.Path
    :param dir_audio: Directory of the audio files for the split.
    :type dir_audio: pathlib.Path
    :param words_list: List of the words.
    :type words_list: list[str]
    :param chars_list: List of the characters.
    :type chars_list: list[str]
    :param settings_ann: Settings for the annotations.
    :type settings_ann: dict
    :param settings_audio: Settings for the audio.
    :type settings_audio: dict
    :param settings_output: Settings for the output files.
    :type settings_output: dict
    """
    # Make sure that the directory exists
    dir_split.mkdir(parents=True, exist_ok=True)

    captions_fields = [settings_ann['captions_fields_prefix'].format(i)
                       for i in range(1, int(settings_ann['nb_captions']) + 1)]

    # For each sound:
    for csv_entry in csv_split:
        file_name_audio = csv_entry[settings_ann['audio_file_column']]

        audio = load_audio_file(
            audio_file=str(dir_audio.joinpath(file_name_audio)),
            sr=int(settings_audio['sr']),
            mono=settings_audio['to_mono'])

        for caption_ind, caption_field in enumerate(captions_fields):
            caption = csv_entry[caption_field]

            words_caption = get_sentence_words(
                caption,
                unique=settings_ann['use_unique_words_per_caption'],
                keep_case=settings_ann['keep_case'],
                remove_punctuation=settings_ann['remove_punctuation_words'],
                remove_specials=not settings_ann['use_special_tokens'])

            chars_caption = list(chain.from_iterable(
                clean_sentence(
                    caption,
                    keep_case=settings_ann['keep_case'],
                    remove_punctuation=settings_ann['remove_punctuation_chars'],
                    remove_specials=True)))

            if settings_ann['use_special_tokens']:
                chars_caption.insert(0, ' ')
                chars_caption.insert(0, '<sos>')
                chars_caption.append(' ')
                chars_caption.append('<eos>')

            indices_words = [words_list.index(word) for word in words_caption]
            indices_chars = [chars_list.index(char) for char in chars_caption]

            # create the numpy object with all elements
            np_rec_array = np.rec.array(np.array(
                (file_name_audio, audio, caption, caption_ind,
                 np.array(indices_words), np.array(indices_chars)),
                dtype=[
                    ('file_name', f'U{len(file_name_audio)}'),
                    ('audio_data', np.dtype(object)),
                    ('caption', f'U{len(caption)}'),
                    ('caption_ind', 'i4'),
                    ('words_ind', np.dtype
                    (object)),
                    ('chars_ind', np.dtype(object))
                ]))

            #  save the numpy object to disk
            dump_numpy_object(
                np_obj=np_rec_array,
                file_name=dir_split.joinpath(
                    settings_output['files']['np_file_name_template'].format(
                        audio_file_name=file_name_audio,
                        caption_index=caption_ind)))


def get_amount_of_file_in_dir(the_dir: Path) \
        -> int:
    """Counts the amount of files in a directory.

    :param the_dir: Directory.
    :type the_dir: pathlib.Path
    :return: Amount of files in directory.
    :rtype: int
    """
    counter = count()

    deque(zip(the_dir.iterdir(), counter))

    return next(counter)


def get_annotations_files(settings_ann: MutableMapping[str, Any],
                          dir_ann: Path)\
        -> Tuple[List[MutableMapping[str, Any]], List[MutableMapping[str, Any]]]:
    """Reads, process (if necessary), and returns tha annotations files.

    :param settings_ann: Settings to be used.
    :type settings_ann: dict
    :param dir_ann: Directory of the annotations files.
    :type dir_ann: pathlib.Path
    :return: Development and evaluation annotations files.
    :rtype: list[collections.OrderedDict], list[collections.OrderedDict]
    """
    field_caption = settings_ann['captions_fields_prefix']
    csv_development = read_csv_file(
        file_name=settings_ann['development_file'],
        base_dir=dir_ann)
    csv_evaluation = read_csv_file(
        file_name=settings_ann['evaluation_file'],
        base_dir=dir_ann)

    caption_fields = [field_caption.format(c_ind) for c_ind in range(1, 6)]

    for csv_entry in chain(csv_development, csv_evaluation):
        # Clean sentence to remove any spaces before punctuations.

        captions = [clean_sentence(
            csv_entry.get(caption_field),
            keep_case=True,
            remove_punctuation=False,
            remove_specials=False)
            for caption_field in caption_fields]

        if settings_ann['use_special_tokens']:
            captions = [f'<SOS> {caption} <EOS>' for caption in captions]

        [csv_entry.update({caption_field: caption})
         for caption_field, caption in zip(caption_fields, captions)]

    return csv_development, csv_evaluation

# EOF
