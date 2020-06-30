"""
Generate the word list used in tagging pre-training.
"""
import pickle


def dict_add(word_freq_dict, w, f):
    if w in word_freq_dict.keys():
        word_freq_dict[w] += f
    else:
        word_freq_dict[w] = f
    return word_freq_dict


# load data
word_freq = pickle.load(open(r'../data/pickles/words_frequencies.p', 'rb'))
word_list = pickle.load(open(r'../data/pickles/words_list.p', 'rb'))

word_freq = [(w, f) for w, f in zip(word_list, word_freq)]
word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

# Save a complete word list that contains words occur more than twice
word_list_short = [w for w, f in word_freq if f > 2]

stop_words = []
stop_words.extend(word_freq[:20])  # Exclude the top 20 words, save in list
word_freq = word_freq[20:]

word_freq = [(w, f) for w, f in word_freq if len(w) > 2]  # Exclude words that have no more than 2 letters.

sorted([(w, f / len(w)) for w, f in word_freq if len(w) < 5], key=lambda x: x[1], reverse=True)  # 查看短字母数量的词，并以单词长度反向加权

for w, f in sorted([(w, f / len(w)) for w, f in word_freq if len(w) < 5], key=lambda x: x[1], reverse=True)[:20]:
    # [(w, f / len(w)) for w, f in word_freq if len(w) < 5]:
    # First, weight each word frequency (that has less than 5 letters) inverse proportional to the number of letters
    # by dividing the word frequency by the number of letters.
    # sorted(..., key=lambda x: x[1], reverse=True)[:20]
    # Then, sort the weighted list and add the top 20 words to the stop words list.
    stop_words.extend([(w, int(len(w) * f))])

# Exclude the words that are in the stop word list.
word_freq = [(w, f) for w, f in word_freq if w not in [s[0] for s in stop_words]]

# Merge the word to its original form:
word_freq_dict = {}
for w, f in word_freq:
    if w[-3:] == 'ing' and w[:-3] in word_list:  # playing
        word_freq_dict = dict_add(word_freq_dict, w[:-3], f)
    elif w[-3:] == 'ing' and w[:-4] in word_list:  # running
        word_freq_dict = dict_add(word_freq_dict, w[:-4], f)
    elif w[-2:] == 'ly' and w[:-2] in word_list:  # slowly
        word_freq_dict = dict_add(word_freq_dict, w[:-2], f)
    elif w[-2:] == 'ly' and w[:-3] in word_list:  # lly
        word_freq_dict = dict_add(word_freq_dict, w[:-3], f)
    elif w[-1] == 's' and w[:-1] in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-1], f)
    elif w[-2:] == 'es' and w[:-2] in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-2], f)
    elif w[-1] == 'd' and w[:-1] in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-1], f)
    elif w[-2:] == 'ed' and w[:-2] in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-2], f)
    else:
        word_freq_dict = dict_add(word_freq_dict, w, f)

# Check if there is any word in stop word list that has third person singular,
# in which case these words should be added to the word list.
for w, f in stop_words:
    if w + 's' in word_list_short or w + 'es' in word_list_short:
        word_freq_dict = dict_add(word_freq_dict, w, f)
    elif w[-1] == 's' and w[:-1] in word_list_short:
        word_freq_dict = dict_add(word_freq_dict, w[:-1], f)
    elif w[-2:] == 'es' and w[:-2] in word_list_short:
        word_freq_dict = dict_add(word_freq_dict, w[:-2], f)

# sort according to frequency
word_freq = sorted([(w, f) for w, f in word_freq_dict.items()], key=lambda x: x[1], reverse=True)

# Exclude words that have no more than 2 letters.
# (Because here the words like "being" will be transformed into "be" in the previous steps)
word_freq = [(w, f) for w, f in word_freq if len(w) > 2]

word_list_final = [w for w, f in word_freq][:300]  # Use top 300 words.

pickle.dump(word_list_final, open('word_list_pretrain_rules.p', 'wb'))
