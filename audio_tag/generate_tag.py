import csv
import pickle
from .captions_functions import clean_sentence,get_words_counter,get_sentence_words
import copy

with open('word_list_pretrain_rules.p', 'rb') as f:
    testtag = pickle.load(f)

def c_wordList(x):
    WordList = copy.copy(x)
    for w in x:
        if w[-3:] == 'ing' and w[:-3] in testtag: #playing
            WordList.remove(w)
            WordList.add(w[:-3])
        elif w[-3:] == 'ing' and w[:-4] in testtag: #running
            WordList.remove(w)
            WordList.add(w[:-4])
        elif w[-2:] == 'ly' and w[:-2] in testtag: # slowly
            WordList.remove(w)
            WordList.add(w[:-2])
        elif w[-2:] == 'ly' and w[:-3] in testtag: # lly
            WordList.remove(w)
            WordList.add(w[:-3])
        elif w[-1] == 's' and w[:-1] in testtag:
            WordList.remove(w)
            WordList.add(w[:-1])
        elif w[-2:] == 'es' and w[:-2] in testtag:
            WordList.remove(w)
            WordList.add(w[:-2])
        elif w[-1] == 'd' and w[:-1] in testtag:
            WordList.remove(w)
            WordList.add(w[:-1])
        elif w[-2:] == 'ed' and w[:-2] in testtag:
            WordList.remove(w)
            WordList.add(w[:-2])
    return WordList

def gen_tag(split):
    audioAttributes = [] 
    allWordList = []
    with open('clotho_captions_{}.csv'.format(split), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        file_names = [row['file_name'] for row in reader]
    with open('clotho_captions_{}.csv'.format(split), 'r') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            curAttributes = []  
            count += 1  
            if count == 1:
                continue
            curCaptionList = row[1:]  
            curWordList = []
            for caption in curCaptionList:
                curWordList.extend(get_sentence_words(caption))
            curWordList = c_wordList(set(curWordList))
            allWordList.append(list(curWordList))

    allAttributes = {}
    for file_name, audioList in zip(file_names, allWordList):
        curAttributes = []
        for attribute in testtag:
            if attribute in audioList:
                curAttributes.append(attribute)
        allAttributes[file_name] = curAttributes

    allAttributesNum = {}
    for file_name, audioList in zip(file_names, allWordList):
        curAttributes = [0 for s in range(len(testtag))]
        for i, attribute in enumerate(testtag):
            if attribute in audioList:
                curAttributes[i] = 1
        allAttributesNum[file_name] = curAttributes

    with open('audioTagNum_{}_fin.pickle'.format(split), 'wb') as f:
        pickle.dump(allAttributesNum, f)

if __name__ == '__main__':
    gen_tag('development')
    gen_tag('evaluation')


