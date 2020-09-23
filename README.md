## Audio Captioning based on Transformer and Pre-training CNN

### Setting up the Code and Environment

1. Clone this repository: `git clone https://github.com/lukewys/dcase_2020_T6.git`
2. Install pytorch >=1.4.0
3. Use  pip to install dependencies: `pip install -r requirements.txt`

### Preparing the Data

- Download the required dataset  of  DCASE2020 Automated Audio Captioning task

- Enter the `create_dataset` directory，Put Clotho data set into `data` folder structured like this:

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

- Enter the `create_dataset` directory.

- Run `main.py` to create dataset and extract features under current directory. Result folders would be `data_splits`, `data_splits_audio` and `pickles`. (The progress bar is not behaving correctly, please wait until the program to finish)

- Then, run `create_test_set.py` to create test set under current directory. Result folder would be `test_data`.

### Setup COCO caption

Please setup coco caption and download the file needed according to https://github.com/audio-captioning/caption-evaluation-tools. (~300MB of files will be downloaded)

### Pretrain CNN

- Enter the `audio_tag` directory.
- First, run`generate_word_list.py` to generate word list `word_list_pretrain_rules.p`, then run `generate_tag.py` to generate the tag label: `audioTagNum_development_fin.pickle`  and `audioTagNum_evaluation_fin.pickle`.
- Back to the project directory, run `Tag_train.py` to train the model. The model is saved in `./models/tag_models`.



### Pretrain Word Embedding

- Enter `the word_embedding_pretrain `  directory, run `word2vec.py`



### Run the Project

- The file hparams.py holds the default arguments for our project. You can modify these parameters according to your own needs. 

- You can use our pretrained models in `models/best.pt`  to  reproduce our results.

#### train

Set `mode=train` in hparams.py. Then you can run the project using the set script:

```python
#You can modify the running script as needed
python run.py
```

The trained models and logs will be saved in `models/{hparams.name}`.

#### eval

set `mode=eval` in hparams.py. Then choose the model you want to evaluate,run

`python train.py`. The system will score the model based on the evaluation dataset.

#### test

set `mode=test` in hparams.py. Then choose the model you want to test,run `python train.py`. The system will generate the corresponding captions according to the test dataset.



## TODO

how to get the results and score

how to load trained weight and reproduce exact result submitted in challenge

how to get all the scores in ablation test
