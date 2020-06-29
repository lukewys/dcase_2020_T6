import os
import glob
import librosa
import numpy as np

from tools.features_log_mel_bands import feature_extraction

from tools.file_io import load_yaml_file
from tools.argument_parsing import get_argument_parser
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
from functools import partial

executor = ProcessPoolExecutor(max_workers=cpu_count())


def wav_to_mel(wav_file, settings):
    output_dir = r'./data/test_data'
    os.makedirs(output_dir, exist_ok=True)

    settings_audio = settings['dataset_creation_settings']['audio']
    settings_features = settings['feature_extraction_settings']

    y = librosa.load(path=wav_file, sr=int(settings_audio['sr']), mono=settings_audio['to_mono'])[0]

    mel = feature_extraction(y, **settings_features['process'])

    file_name = os.path.splitext(os.path.basename(wav_file))[0]

    np.save(f'{output_dir}/{file_name}.npy', mel)


if __name__ == '__main__':
    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose
    settings = load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    file_list = glob.glob('./data/clotho_test_audio/*.wav')

    print('Create test data in multiprocess:')

    futures = []
    for file in file_list:
        futures.append(executor.submit(
            partial(wav_to_mel, file, settings)))
    [future.result() for future in tqdm(futures)]

    print('Test data created.')

