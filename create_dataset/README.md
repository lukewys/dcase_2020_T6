Put Clotho data set into `data` folder structured like this:

```
data
├── clotho_csv_files
│   ├── clotho_captions_development.csv
│   ├── clotho_captions_evaluation.csv
│   ├── clotho_metadata_development.csv
│   └── clotho_metadata_evaluation.csv
├── clotho_audio_files
│   ├── development
│   └── evaluation
└── clotho_test_audio
│   ├── test_0001.wav
│   ├── test_0002.wav
│   ├── ...
│   ├── test_1043.wav

```



Run `main.py` to create dataset and extract features under current directory. Result folders would be `data_splits`, `data_splits_audio` and `pickles`. (The progress bar is not behaving correctly, please wait until the program to finish)

Then, run `create_test_set.py` to create test set under current directory. Result folder would be `test_data`.