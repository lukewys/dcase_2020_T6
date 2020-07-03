from pathlib import Path


class hparams:
    batch_size = 16
    nhead = 4
    nhid = 192
    nlayers = 2
    ninp = 64
    ntoken = 4367 + 1
    clip_grad = 2.5
    lr = 3e-4  # learning rate
    beam_width = 3
    training_epochs = 50
    log_interval = 100
    checkpoint_save_interval = 5

    seed = 1111
    device = 'cuda:0'   #'cuda:0' 'cuda:1' 'cpu'
    mode = 'train'

    name = 'base'

    nkeyword = 4979

    label_smoothing = True
    load_pretrain_cnn = True
    load_pretrain_emb = False
    load_pretrain_model = False
    spec_augmentation = True
    scheduler_decay = 0.98

    # data(default)
    data_dir = Path(r'./create_dataset/data/data_splits')
    eval_data_dir = r'./create_dataset/data/data_splits/evaluation'
    train_data_dir = r'./create_dataset/data/data_splits/development'
    test_data_dir = r'./create_dataset/data/test_data'
    word_dict_pickle_path = r'./create_dataset/data/pickles/words_list.p'
    word_freq_pickle_path = r'./create_dataset/data/pickles/words_frequencies.p'


    # pretrain_model
    pretrain_emb_path = r'models/w2v_192.mod'
    pretrain_cnn_path = r'models/TagModel_60.pt'
    pretrain_model_path = r'models/20.pt'
