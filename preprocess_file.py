# -*- coding: utf-8 -*-

import re

import pickle

def generate_lexicon():
    with open('dataset/MPQA/lexicon_negpos','rb') as fd:
        dict_MPQA_lexicon = pickle.load(fd)

    with open('dataset/sst/lexicon_negpos','rb') as fd:
        dict_sst_lexicon = pickle.load(fd)

    dict_lexicon = {}
    for key,value in dict_MPQA_lexicon.items():
        if(key in dict_sst_lexicon and value != dict_sst_lexicon[key]):
            continue
        dict_lexicon[key] = value

    for key,value in dict_sst_lexicon.items():
        if(key in dict_MPQA_lexicon and value != dict_MPQA_lexicon[key]):
            continue
        dict_lexicon[key] = value


    print('len of lexicon: {}'.format(len(dict_lexicon)))
    with open('dataset/lexicon_negpos','wb') as fd:
        pickle.dump(dict_lexicon,fd)

    # print(len(dict_sst_lexicon))
    # print(len(dict_MPQA_lexicon))
    # print(len(dict_lexicon))
    return dict_lexicon

def generate_MPQA_lexicon():
    dict_MPQA_lexicon = {}
    with open('dataset/MPQA/subjclueslen1-HLTEMNLP05.tff','r') as fd:
        for line_id,line in enumerate(fd):
            if(len(line)<4):
                continue
            lst_line = line.split()
            word = lst_line[2].split('=')[1]
            label = lst_line[5].split('=')[1]

            if(label=='negative'):
                label = 0
            elif(label=='positive'):
                label = 1
            else:#neural
                continue
            dict_MPQA_lexicon[word] = label
    with open('dataset/MPQA/lexicon_negpos','wb') as fd:
        pickle.dump(dict_MPQA_lexicon, fd)

    print('len of MPQA lexicon: {}'.format(len(dict_MPQA_lexicon)))
    return dict_MPQA_lexicon

def generate_sst_lexicon():
    with open('dataset/sst/dictionary','rb') as fd:
        dict_phrase2index = pickle.load(fd)
    with open('dataset/sst/datasetSentences','rb') as fd:
        dict_index2sentence = pickle.load(fd)
    with open('dataset/sst/sentiment_labels','rb') as fd:
        dict_phraseindex2label = pickle.load(fd)

    dict_word2label = {}
    dict_word2posneg = {}
    for sentence in dict_index2sentence.values():
        lst_line = sentence.split()
        for word in lst_line:
            if word not in dict_word2label:
                label = dict_phraseindex2label[dict_phrase2index[word]]
                dict_word2label[word] = label

    for word,label in dict_word2label.items():
        if(label<=0.4):
            dict_word2posneg[word] = 0
        elif(label > 0.6):
            dict_word2posneg[word] = 1

    with open('dataset/sst/lexicon','wb') as fd:
        pickle.dump(dict_word2label, fd)

    with open('dataset/sst/lexicon_negpos','wb') as fd:
        pickle.dump(dict_word2posneg, fd)

    print(len(dict_word2posneg))



def generate_sst_sentence_label():
    dict_phrase2index = {}
    dict_index2sentence = {}
    dict_sentence2index = {}
    dict_phraseindex2label = {}
    dict_sentindex2label = {}
    with open('dataset/sst/dictionary','rb') as fd:
        dict_phrase2index = pickle.load(fd)

    with open('dataset/sst/datasetSentences','rb') as fd:
        dict_index2sentence = pickle.load(fd)
        dict_sentence2index = pickle.load(fd)

    with open('dataset/sst/sentiment_labels','rb') as fd:
        dict_phraseindex2label = pickle.load(fd)

    for sentindex,sentence in dict_index2sentence.items():
        try:
            phraseindex = dict_phrase2index[sentence]
            label = dict_phraseindex2label[phraseindex]
            dict_sentindex2label[sentindex] = label
        except:
            print(sentindex)

    with open('dataset/sst/sentences_labels','wb') as fd:
        pickle.dump(dict_sentindex2label,fd)

    dict_sentindex2label5 = {}
    for sentindex,label in dict_sentindex2label.items():
        if(label<=0.2):
            dict_sentindex2label5[sentindex] = 0
        elif(label <=0.4):
            dict_sentindex2label5[sentindex] = 1
        elif(label <= 0.6):
            dict_sentindex2label5[sentindex] = 2
        elif(label <= 0.8):
            dict_sentindex2label5[sentindex] = 3
        else:
            dict_sentindex2label5[sentindex] = 4

    with open('dataset/sst/sentences_labels5', 'wb') as fd:
        pickle.dump(dict_sentindex2label5, fd)

    return dict_sentindex2label


def process_sst_dictionary():
    dict = {}
    with open('dataset/sst/dictionary1.txt','r') as fd:
        for line_id, line in enumerate(fd):
            if len(line)>1:
                line = line[:-1]
                lst_line = line.split('|')
                dict[lst_line[0]] = int(lst_line[1])

    with open('dataset/sst/dictionary','wb') as fd:
        pickle.dump(dict, fd)

    return dict

def process_sst_datasetSentences():
    dict_index2sentence = {}
    dict_sentence2index = {}
    with open('dataset/sst/datasetSentences1.txt','rt') as fd:
        for line_id, line in enumerate(fd):
            if(line_id == 0):
                continue
            if(len(line) > 1):
                line = line[:-1]
                lst_line = line.split('\t')
                index = int(lst_line[0])
                sentence = lst_line[1]
                dict_index2sentence[index] = sentence
                dict_sentence2index[sentence] = index

    with open('dataset/sst/datasetSentences','wb') as fd:
        pickle.dump(dict_index2sentence, fd)
        pickle.dump(dict_sentence2index, fd)

    return dict_index2sentence,dict_sentence2index

def process_datasetSplit():
    dict_index2split = {}
    dict_split2indexset = {}

    with open('dataset/sst/datasetSplit.txt','r') as fd:
        for line_id,line in enumerate(fd):
            if(line_id == 0):
                continue
            if(len(line) > 1):
                line = line[:-1]
                lst_line = line.split(',')
                index = int(lst_line[0])
                split_id = int(lst_line[1])
                dict_index2split[index] = split_id
                if(split_id not in dict_split2indexset):
                    dict_split2indexset[split_id] = [index]
                else:
                    dict_split2indexset[split_id].append(index)

    with open('dataset/sst/datasetSplit','wb') as fd:
        pickle.dump(dict_index2split,fd)
        pickle.dump(dict_split2indexset,fd)

    return dict_index2split,dict_split2indexset

def process_sentiment_labels():
    dict_phraseindex2label = {}
    with open('dataset/sst/sentiment_labels.txt','r') as fd:
        for line_id, line in enumerate(fd):
            if(line_id == 0):
                continue
            if(len(line) > 1):
                line = line[:-1]
                lst_line = line.split('|')
                phrase_id = int(lst_line[0])
                label = float(lst_line[1])
                dict_phraseindex2label[phrase_id] = label

    with open('dataset/sst/sentiment_labels','wb') as fd:
        pickle.dump(dict_phraseindex2label, fd)

    return dict_phraseindex2label

def preprocess_files():
    fd1 =  open('dataset/sst/datasetSentences1.txt','wt')
    with open('dataset/sst/datasetSentences.txt','r') as fd:
        for line_id,line in enumerate(fd):
            if(line_id!=0):
                line = line.replace('-LRB-', '(')
                line = line.replace('-RRB-', ')')
                line = line.replace('Â', '')
                line = line.replace('Ã©', 'e')
                line = line.replace('Ã¨', 'e')
                line = line.replace('Ã¯', 'i')
                line = line.replace('Ã³', 'o')
                line = line.replace('Ã´', 'o')
                line = line.replace('Ã¶', 'o')
                line = line.replace('Ã±', 'n')
                line = line.replace('Ã ', 'a')
                line = line.replace('Ã¡', 'a')
                line = line.replace('Ã¢', 'a')
                line = line.replace('Ã£', 'a')
                line = line.replace('\xc3\x83\xc2\xa0', 'a')
                line = line.replace('Ã¼', 'u')
                line = line.replace('Ã»', 'u')
                line = line.replace('Ã§', 'c')
                line = line.replace('Ã¦', 'ae')
                line = line.replace('Ã­', 'i')
                line = line.replace('\xa0', ' ')
                line = line.replace('\xc2', '')
            fd1.write(line)
    fd.close()

    fd2 = open('dataset/sst/dictionary1.txt','wt')
    with open('dataset/sst/dictionary.txt','r') as fd:
        for line_id,line in enumerate(fd):
            line = line.replace('é', 'e')
            line = line.replace('è', 'e')
            line = line.replace('ï', 'i')
            line = line.replace('í', 'i')
            line = line.replace('ó', 'o')
            line = line.replace('ô', 'o')
            line = line.replace('ö', 'o')
            line = line.replace('á', 'a')
            line = line.replace('â', 'a')
            line = line.replace('ã', 'a')
            line = line.replace('à', 'a')
            line = line.replace('ü', 'u')
            line = line.replace('û', 'u')
            line = line.replace('ñ', 'n')
            line = line.replace('ç', 'c')
            line = line.replace('æ', 'ae')
            line = line.replace('\xa0', ' ')
            line = line.replace('\xc2', '')

            fd2.write(line)
    fd2.close()

def process_MR_files():
    fd1 = open('dataset/MR/rt-polarity.neg1','wt')
    fd2 = open('dataset/MR/rt-polarity.pos1','wt')

    space_token = re.compile(r' +')

    with open('dataset/MR/rt-polarity.neg-utf8','rt') as fd:
        for line_id,line in enumerate(fd):
            line = line.replace('é', 'e')
            line = line.replace('è', 'e')
            line = line.replace('ï', 'i')
            line = line.replace('í', 'i')
            line = line.replace('ó', 'o')
            line = line.replace('ô', 'o')
            line = line.replace('ö', 'o')
            line = line.replace('á', 'a')
            line = line.replace('â', 'a')
            line = line.replace('ã', 'a')
            line = line.replace('à', 'a')
            line = line.replace('ü', 'u')
            line = line.replace('û', 'u')
            line = line.replace('ñ', 'n')
            line = line.replace('ç', 'c')
            line = line.replace('æ', 'ae')
            line = line.replace('\xa0', ' ')
            line = line.replace('\xc2', '')

            line = line.replace('‘', '\'')
            line = line.replace('’', '\'')
            line = line.replace('\'',' \' ')
            line = line.replace('\' s ',' \'s ')
            line = line.replace('n \' t ', ' n\'t ')
            line = line.replace('\' ve ', ' \'ve ')
            line = line.replace('\' ll ', ' \'ll ')
            line = line.replace('\' re ', ' \'re ')

            line = line.replace('[', ' [ ')
            line = line.replace(']', ' ] ')

            # 去掉多余空格
            line = space_token.sub(' ', line)
            # 去掉末尾的空格
            if (len(line) > 2 and line[-2] == ' '):
                line = line[:-2] + '\n'

            fd1.write(line)
    fd1.close()

    with open('dataset/MR/rt-polarity.pos-utf8','rt') as fd:
        for line_id,line in enumerate(fd):
            line = line.replace('é', 'e')
            line = line.replace('è', 'e')
            line = line.replace('ï', 'i')
            line = line.replace('í', 'i')
            line = line.replace('ó', 'o')
            line = line.replace('ô', 'o')
            line = line.replace('ö', 'o')
            line = line.replace('á', 'a')
            line = line.replace('â', 'a')
            line = line.replace('ã', 'a')
            line = line.replace('à', 'a')
            line = line.replace('ü', 'u')
            line = line.replace('û', 'u')
            line = line.replace('ñ', 'n')
            line = line.replace('ç', 'c')
            line = line.replace('æ', 'ae')
            line = line.replace('\xa0', ' ')
            line = line.replace('\xc2', '')

            line = line.replace('‘', '\'')
            line = line.replace('’', '\'')
            line = line.replace('\'',' \' ')
            line = line.replace('\' s ',' \'s ')
            line = line.replace('\' d ', ' \'d ')
            line = line.replace('n \' t ', ' n\'t ')
            line = line.replace('\' ve ', ' \'ve ')
            line = line.replace('\' ll ', ' \'ll ')
            line = line.replace('\' re ', ' \'re ')


            line = line.replace('[', ' [ ')
            line = line.replace(']', ' ] ')

            # 去掉多余空格
            line = space_token.sub(' ', line)
            # 去掉末尾的空格
            if (len(line) > 2 and line[-2] == ' '):
                line = line[:-2] + '\n'

            fd2.write(line)
    fd2.close()



# preprocess_files()
# dict = process_sst_dictionary()
# print(len(dict))
#
# dict_index2sentence,dict_sentence2index = process_sst_datasetSentences()
# dict_index2split,dict_split2indexset = process_datasetSplit()
# dict_phraseindex2label = process_sentiment_labels()
# dict_sentindex2label = generate_sst_sentence_label()
# generate_sst_lexicon()
# generate_MPQA_lexicon()
# generate_lexicon()
# process_MR_files()