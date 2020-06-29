import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from audio_tag.data_loader import tag_loader
import numpy as np
from hparams import hparams as hp
from encoder import Tag
from tqdm import tqdm

class_num = 300
device = torch.device("cuda:1" if hp.cuda else "cpu")
data_dir = hp.data_dir
learning_rate=1e-3
model = Tag(class_num).to(device)
training_data = tag_loader(data_dir=data_dir, split='development',
                                      batch_size=16,class_num=class_num)
test_data = tag_loader(data_dir=data_dir, split='evaluation',
                               batch_size=16,class_num=class_num)
optimizer =torch.optim.Adam(model.parameters(), lr=learning_rate,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
# optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
tag_loss = nn.BCELoss()

def train(epoch):
    # bar  = tqdm(training_data,total=len(training_data))
    loss_list = []
    model.train()
    with tqdm(training_data,total=len(training_data)) as bar:
        for i, (feature, tag) in enumerate(bar):
            feature = feature.to(device)
            tag = tag.to(device)
            optimizer.zero_grad()
            out_tag = model(feature)
            loss = tag_loss(out_tag,tag)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch, i, np.mean(loss_list)))
    return np.mean(loss_list)

def test(epoch):
    eva_loss = []
    model.eval()
    with torch.no_grad():
        for i, (feature, tag) in enumerate(test_data):
            feature = feature.to(device)
            tag = tag.to(device)
            out_tag = model(feature)
            loss = tag_loss(out_tag, tag)
            eva_loss.append(loss.item())
    mean_loss = np.mean(eva_loss)
    print("epoch:{:d}--testloss:{:.6f}".format(epoch,mean_loss.item()))

    # return  mean_loss


if __name__ == '__main__':
    train_b = True
    epoch_last = 0
    if train_b:
        # model.load_state_dict(torch.load("./models/280/TagModel_{}.pt".format(str(40))))
        for epoch in range(epoch_last+1,epoch_last+5):
            train(epoch)
            scheduler.step(epoch)
            test(epoch)
            if epoch%5==0:
                torch.save(model.state_dict(),'./models/tag_models/TagModel_{}.pt'.format(epoch))
    else:
        for epoch in range(0,225,5):
            model.load_state_dict(torch.load("./models/tag_models/TagModel_{}.pt".format(str(epoch))))
            test(epoch)



