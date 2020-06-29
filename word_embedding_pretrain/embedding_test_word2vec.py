from gensim.models.word2vec import LineSentence, Word2Vec

model = Word2Vec.load("./w2v.mod")


def check_similar_word(word, topn=5):
    similar_word_dict = {}
    for key in model.wv.similar_by_word(word, topn=topn):
        similar_word_dict[key[0]] = key[1]
    return similar_word_dict


print('rain:')
print(check_similar_word('rain'))

print('vehicle:')
print(check_similar_word('vehicle'))

print('people:')
print(check_similar_word('people'))
