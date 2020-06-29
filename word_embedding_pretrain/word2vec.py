# -*- coding: utf-8 -*-

import re

import codecs
import logging
from gensim.models.word2vec import LineSentence, Word2Vec
import sys
import pandas as pd
import imp

imp.reload(sys)

origin_file = "../create_dataset/data/clotho_csv_files/clotho_captions_development.csv"
output_file = "./target_development.txt"
rd = '[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、：:~@#￥%……&*（）《》<>]+'

print("prepare data")
ori = codecs.open(origin_file, 'r', encoding='utf-8')
out = codecs.open(output_file, 'w', encoding='utf-8')
print("read files")

text = pd.read_csv(origin_file, index_col=None)[
    ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.tolist()

out_text = [re.sub(rd, ' ', t.lower()) for caption in text for t in caption]
out_text = '\n'.join(out_text)
out.writelines(out_text)

ori.close()
out.close()

print("Generated word vector")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = LineSentence(output_file)
model = Word2Vec(sentences, size=192, min_count=1, iter=1000, workers=8)
model.train(sentences, total_examples=model.corpus_count, epochs=1000)
model.save("./w2v_192.mod")
print("Done")
