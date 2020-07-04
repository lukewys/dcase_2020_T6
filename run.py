import os
import argparse

# trained by Tag_train.py
tag_model_path = r'models/TagModel_60.pt'
#fine tune
# caption_model_path ='models/18.pt'


#base
lr = 3e-4
training_epochs = 30
name = 'base'
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name}')

#smoothing
lr = 3e-4
training_epochs = 30
name = 'smoothing'
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name}  --label_smoothing')

#augmentation+smoothing
lr = 3e-4
training_epochs = 30
name = 'augmentation+smoothing'
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name} --spec_augmentation --label_smoothing')


#augmentation+smoothing+pretrain_cnn+freeze_cnn
lr = 3e-4
training_epochs = 30
name = 'augmentation+smoothing+pretrain_cnn+freeze_cnn '
os.system(f'python train.py  --training_epochs {training_epochs}  --lr {lr} '
          f'--name {name} --spec_augmentation --label_smoothing '
          f'--load_pretrain_cnn --freeze_cnn')

#fine-tune
 lr = 1e-4
training_epochs = 30
scheduler_decay = 0.98
caption_model_path ='models/augmentation+smoothing+pretrain_cnn+freeze_cnn/20.pt'
name = 'fine-tune'
os.system(f'python train.py --lr {lr} --scheduler_decay {scheduler_decay} '
          f'--training_epochs {training_epochs} --name {name} '
          f'--load_pretrain_model --pretrain_model_path {caption_model_path} '
          f'--spec_augmentation --label_smoothing ')         