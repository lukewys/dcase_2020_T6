import os
import argparse

# trained by Tag_train.py
tag_model_path = r'models/TagModel_60.pt'


#base
lr = 3e-4
training_epochs = 30
name = 'base'
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name}')

#augmentation
lr = 3e-4
training_epochs = 30
name = 'augmentation'
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name} --spec_augmentation ')

#augmentation+smoothing
lr = 3e-4
training_epochs = 30
name = 'augmentation+smoothing'
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name} --spec_augmentation --label_smoothing')

#augmentation+smoothing+pretrain_cnn
lr = 3e-4
training_epochs = 30
name = 'augmentation+smoothing+pretrain_cnn '
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name} --spec_augmentation --label_smoothing '
          f'--load_pretrain_cnn')


#augmentation+smoothing+pretrain_cnn+pretrain_emb
lr = 3e-4
training_epochs = 30
name = 'augmentation+smoothing+pretrain_cnn+pretrain_emb'
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name} --spec_augmentation --label_smoothing '
          f'--load_pretrain_cnn --load_pretrain_emb ')

#Select the model with the highest score above for fine-tuning
lr = 1e-4
training_epochs = 20
scheduler_decay = 1.0
#fine tune
caption_model_path ='models/18.pt'
name = 'fine_tune'
os.system(f'python train.py --lr {lr} --scheduler_decay {scheduler_decay} '
          f'--training_epochs {training_epochs} --name {name} '
          f'--load_pretrain_model --pretrain_model_path {caption_model_path} ')