from util import align_word_embedding
from gensim.models.word2vec import LineSentence, Word2Vec
from util import get_word_dict

if __name__ == '__main__':
    word_dict_pickle_path = r'../create_dataset/data/pickles/words_list.p'
    w2v_model_path = './w2v.mod'
    word_dict = get_word_dict(word_dict_pickle_path)
    word_emb = align_word_embedding(word_dict_pickle_path, w2v_model_path, 4367 + 1, 256)
