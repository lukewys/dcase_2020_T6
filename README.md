## Audio Captioning based on Transformer and Pre-training CNN

### Setting up the Code and Environment

1. Clone this repository: `git clone https://github.com/lukewys/dcase_2020_T6.git`
2. Install pytorch >=1.4.0
3. Use  pip to install dependencies: `pip install -r requirements.txt`

### Preparing the Data

- [Download](http://dcase.community/challenge2020/task-automatic-audio-captioning#download) the required dataset of DCASE2020 Automated Audio Captioning task

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



### Cite

If you use our code, pleas kindly cite following:

```
@techreport{wuyusong2020_t6,
    Author = "Wu, Yusong and Chen, Kun and Wang, Ziyue and Zhang, Xuan and Nian, Fudong and Li, Shengchen and Shao, Xi",
    title = "Audio Captioning Based on Transformer and Pre-Training for 2020 {DCASE} Audio Captioning Challenge",
    institution = "DCASE2020 Challenge",
    year = "2020",
    month = "June",
    abstract = "This report proposes an automated audio captioning model for the 2020 DCASE audio captioning challenge. In this challenge, a model is required to be trained from scratch to generate natural language descriptions of a given audio signal. However, as limited data available and restrictions on using pre-trained models trained by external data, training directly from scratch can result in poor performance where acoustic events and language are poorly modeled. For better acoustic event and language modeling, a sequence-to-sequence model is proposed which consists of a CNN encoder and a Transformer decoder. In the proposed model, the encoder and word embedding are firstly pre-trained. Regulations and data augmentations are applied during training, while fine-tuning is applied after training. Experiments show that the proposed model can achieve a SPIDEr score of 0.227 on audio captioning performance."
}
```



## TODO

how to get the results and score

how to load trained weight and reproduce exact result submitted in challenge

how to get all the scores in ablation test
