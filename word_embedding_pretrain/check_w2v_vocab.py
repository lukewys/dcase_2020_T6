from gensim.models.word2vec import LineSentence, Word2Vec
from util import get_word_dict

model = Word2Vec.load("./w2v.mod")
w2v_vocab = [k for k in model.wv.vocab.keys()]
word_dict_pickle_path = r'../create_dataset/data/pickles/words_list.p'
model_vocab = [v for v in get_word_dict(word_dict_pickle_path).values()]
print('model-w2v:')
print(set(model_vocab) - set(w2v_vocab))
print('w2v-model:')
print(set(w2v_vocab) - set(model_vocab))

