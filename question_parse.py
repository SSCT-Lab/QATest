import nltk
import json
from jsonpath import jsonpath
from collections import Counter
import itertools
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import math

pos_dict = {'CC': 0, 'CD': 1, 'DT': 2, 'EX': 3, 'FW': 4, 'IN': 5, 'JJ': 6, 'JJR': 7, 'JJS': 8, 'LS': 9, 'MD': 10,
            'NN': 11, 'NNS': 12, 'NNP': 13, 'NNPS': 14, 'PDT': 15, 'POS': 16, 'PRP': 17, 'PRP$': 18, 'RB': 19,
            'RBR': 20, 'RBS': 21, 'RP': 22, 'TO': 23, 'UH': 24, 'VB': 25, 'VBG': 26, 'VBD': 27, 'VBN': 28, 'VBP': 29,
            'VBZ': 30, 'WDT': 31, 'WP': 32, 'WRB': 33, 'START': 34, 'EOF': 35, '?': 36, '.': 37, '??': 38, 'UNKNOWN': 39
            }
pos_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ',
            'WDT', 'WP', 'WRB', 'START', 'EOF', '?', '.', '??']


def sentence_pos(question):  # 输入一个句子，返回一个 list，对应每个单词的词性
    pos_li = []
    tokens = word_tokenize(question)
    words = nltk.pos_tag(tokens)
    for item in words:
        if item[0] == '?':  # 如果是问号，还是保持问号
            pos_li.append(item[0])
        else:
            pos_li.append(item[1])
    if pos_li[-1] == '?' and pos_li[-2] == '?':  # 双问号结尾的情况，依然保留最后两个问号，但是其他相邻重复的合并
        tmp = ['START'] + [k for k, _ in itertools.groupby(pos_li[:-2])] + ['??'] + ['EOF']
    else:
        tmp = ['START'] + [k for k, _ in itertools.groupby(pos_li)] + ['EOF']

    pos = ['UNKNOWN' if i not in pos_list else i for i in tmp]  # 把列表之外的pos替换为 'UNKNOWN'
    return pos  # return '->'.join(pos)


def get_dtmc_matrix(seeds):  # 输入是当前所有作为种子的数据
    dtmc_matrix = [[0 for i in range(len(pos_dict))] for j in range(len(pos_dict))]
    pos_list = []
    count_set = {}
    for idx in range(len(pos_dict)):
        count_set[idx] = 0

    for seed in seeds:
        pos = sentence_pos(seed['question'])
        for i, j in zip(pos[0::], pos[1::]):
            pos_list.append((i, j))  # 例如 ('START', 'WP') 这样的 tuple
            count_set[pos_dict[i]] += 1

    for item in Counter(pos_list).items():
        start_word_id = pos_dict[item[0][0]]
        sec_word_id = pos_dict[item[0][1]]
        item_count = item[1]/count_set[start_word_id]
        dtmc_matrix[start_word_id][sec_word_id] = item_count

    return dtmc_matrix


# 根据矩阵计算一个句子出现的概率，越小代表训练集中约没见过，越优先选择
def get_sentence_perplexity(data, dtmc_matrix):
    question = data['question']
    pos = sentence_pos(question)
    # print("pos:", pos)
    probability = 1
    for i, j in zip(pos[0::], pos[1::]):
        probability = probability * dtmc_matrix[pos_dict[i]][pos_dict[j]]

    if probability == 0:
        return probability
    else:
        perplexity = math.pow(probability, (-1/len(pos)))
        return perplexity


def get_gram_set(question):  # 输入一个句子，返回它对应的 1-gram，2-gram，3-gram，4-gram的集合
    tokens = word_tokenize(question)
    one_gram = list(ngrams(tokens, 1))
    two_gram = list(ngrams(tokens, 2))
    three_gram = list(ngrams(tokens, 3))
    four_gram = list(ngrams(tokens, 4))
    gram_list = list(set(one_gram).union(two_gram, three_gram, four_gram))
    return set(gram_list)


# 输入是一个扩增句子对应的字典，输出是计算函数结果，越大证明越不相似，需要和所有同源的在种子集合里的句子对比cov的增量
def gram_evaluate(data, seed_set):
    ori_gram_set = set()
    for seed in seed_set:
        if seed['init_q'] == data['init_q']:
            ori_gram_set = ori_gram_set.union(get_gram_set(seed['init_q']))
    aug_gram_set = get_gram_set(data['question'])
    if len(ori_gram_set) == 0:
        return len(aug_gram_set)
    gram_score = len(aug_gram_set-ori_gram_set)/len(ori_gram_set)
    return gram_score


if __name__ == '__main__':
    aug_batch = [
        {'init_q': 'Who was the duke in the battle of Hastings?',
         'question': 'who right was the duke in the victorious battle party of hastings?',
         'answer': ['a', 'a10'], 'is_init': False},
        {'init_q': 'When were the Normans in Normandy?', 'question': 'When were the Normans in Normandy??',
         'answer': ['a', 'a2'], 'is_init': False},
        {'init_q': 'When were the Normans in Normandy?', 'question': 'When were Normans in Normandy?',
         'answer': ['a', 'a2'], 'is_init': True},
        {'init_q': "Who gave their name to Normandy in the 1000's and 1100's",
         'question': "Who gave their name to Normandy in the 1000's and 1100's", 'answer': ['a', 'a6'],
         'is_init': False},
        {'init_q': 'In what country is Normandy located?',
         'question': 'anyway in what country africa is normandy located?', 'answer': ['a', 'a1'], 'is_init': False},
        {'init_q': 'When did the Frankish identity emerge?',
         'question': 'when was did the frankish identity not emerge?', 'answer': ['a', 'a9'], 'is_init': False}]
    test_batch = [
        {'init_q': 'In what country is Normandy located?', 'question': 'In what country is Normandy located?',
         'answer': ['a', 'a1'], 'is_init': True},
        {'init_q': 'When were the Normans in Normandy?', 'question': 'When were the Normans in Normandy?',
         'answer': ['a', 'a2'], 'is_init': True},
        {'init_q': 'From which countries did the Norse originate?',
         'question': 'From which countries did the Norse originate?', 'answer': ['a', 'a3'], 'is_init': True}]

    gram_score = gram_evaluate(aug_batch[2], aug_batch[2:])
    print(gram_score)
    # dtmc_matrix = get_dtmc_matrix(aug_batch)
    # print(dtmc_matrix)
    # pro = get_sentence_perplexity(test_batch[0], dtmc_matrix)
    # print(pro)
