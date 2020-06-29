## Audio Captioning based on Transformer and Pre-training CNN

### Setting up the code

1. Clone this repository.
2. Use  pip to install dependencies.

```
$ git clone https://github.com/lukewys/dcase_2020_T6.git
$ pip install -r requirements_pip.txt
```

### preparing the data

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
  
  ```

- Run `main.py` to create dataset and extract features under current directory. Result folders would be `data_splits`, `data_splits_audio` and `pickles`.

- Then, run `creat_test_set.py` to create test set under current directory. Result folder would be `test_data`.

### pretrain cnn

- Enter the `audio_tag` directory, run `generate_tag.py` to generate the tag file `audioTagNum_development_fin.pickle`  and `audioTagNum_ evaluation_fin.pickle`.
- Back to the project directory, run `Tag_train.py` to train the model. The model is saved in `./models/tag_models`.



### pretrain embedding

- Enter `the word_embedding_pretrain `  directory, run `word2vec.py`



### run the project

You can run the project using the set script:

```python
python run.py
```

  

  



