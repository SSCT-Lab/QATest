import json
from jsonpath import jsonpath
from collections import Counter
import random
import string
import os


def analysis_test_results(file_path):  # 统计 aug 的次数，返回一个字典
    aug = []
    file = open(file_path, encoding='utf-8')
    for line in file.readlines():
        d = json.loads(line)
        if d['eval'] == 0:
            aug.append(d['aug'])
    print(dict(Counter(aug)))


def generate_random_str(randomlength):
    '''
    string.digits = 0123456789
    string.ascii_letters = 26个小写,26个大写
    '''
    str_list = random.sample(string.digits + string.ascii_letters, randomlength)
    random_str = ''.join(str_list)
    return random_str


def generate_random_num(randomlength):
    '''
    string.digits = 0123456789
    '''
    str_list = random.sample(string.digits, randomlength)
    random_str = ''.join(str_list)
    return random_str


if __name__ == '__main__':
    pass
