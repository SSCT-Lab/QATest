import json
from jsonpath import jsonpath
from rouge import Rouge
from question_trans import *
from question_parse import *
from metrics import *
from convert.squad_convert import *
from convert.squad1_convert import *
from convert.race_convert import *
from convert.qca_convert import *
from convert.boolq_convert import *
from convert.webquestions_convert import *
import lib
import random
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

aug_dict = {
    0: 'keybord_mistake',
    1: 'ocr_mistake',
    2: 'spelling_mistake',
    3: 'synonym_replace',
    4: 'adverbial_preposition',
    5: 'insert_word',
    6: 'back_translate',
    7: 'entity_replace',
    8: 'wps',
    9: 'double_question_mark',
}

aug_li = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def random_gen_tests(seeds):
    gen_dataset = []
    for it, seed in enumerate(seeds):
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print("it:", it)
        print(seed['question'])
        tmp_batch, save_batch = [], []
        for i in range(10):
            flag = True
            aug_id = aug_li[i]
            aug_type = aug_dict.get(aug_id)
            aug_question = eval(aug_dict.get(aug_id))(seed['question'])
            if aug_question == seed['question']:  # 扩增失败，生成的问题和原始问题完全一样
                flag = False
            else:
                tmp_batch.append(aug_question)
            for data in tmp_batch:
                if data == seed['init_q'] and rouge1_score(aug_question, data)['r'] == 1:
                    flag = False
            if flag:
                rouge1_f = rouge1_score(aug_question, seed['question'])['f']
                if rouge1_f > 0.5:
                    generate_data = {'init_q': seed['init_q'], 'question': aug_question,
                                     'is_init': False, 'aug_times': 0, 'aug': aug_type}
                    left_key = list(set(seed.keys()) - {'init_q', 'question', 'is_init', 'aug_times', 'aug'})
                    for key in left_key:
                        generate_data[key] = seed[key]
                    save_batch.append(generate_data)
        gen_dataset.extend(save_batch)
    return gen_dataset


def run_drqa(data_type):
    if data_type == "webquestions":
        out_path = "./data/WebQuestions/WebQuestions-test-500_all.txt"
        seed_tests = get_webquestions_data(data_path="./data/WebQuestions/WebQuestions-test-500.txt")
        # seed_tests = seed_tests[:500]
        gen_dataset = random_gen_tests(seed_tests)
        save_webquestions_data(gen_dataset, out_path)
    if data_type == "squad":
        out_path = "./data/squad1.1/SQuAD-v1.1-dev-500_all.json"
        seed_tests = get_squad1_1_data(input="./data/squad1.1/SQuAD-v1.1-dev-500.json")
        # seed_tests = seed_tests[:500]
        gen_dataset = random_gen_tests(seed_tests)
        save_squad1_1_data(gen_dataset, out_path)


def run_albart(data_type):
    if data_type == "squad":
        out_path = "./data/squad/squad-dev-v2.0-500_all.json"
        seed_tests = get_squad_data(input="./data/squad/squad-dev-v2.0-500.json")
        # seed_tests = seed_tests[:500]
        gen_dataset = random_gen_tests(seed_tests)
        save_squad_data(gen_dataset, out_path)
    if data_type == "race":
        out_path = "./data/race/race_test_high-500_all.txt"
        seed_tests = get_race_data(input="./data/race/race_test_high-500.txt")
        # seed_tests = seed_tests[:500]
        gen_dataset = random_gen_tests(seed_tests)
        save_race_data(gen_dataset, out_path)


def run_unifiedqa(data_type):
    if data_type == "boolq":
        out_path_tsv = "./data/boolq/dev-500_all.tsv"
        out_path_jsonl = "./data/boolq/dev-500_all.jsonl"
        seed_tests = get_boolq_data(input="./data/boolq/dev-500.jsonl")
        gen_dataset = random_gen_tests(seed_tests)
        save_boolq_data(gen_dataset, out_path_tsv, out_path_jsonl)


def run_marl():
    out_path = "./data/qca/qca_comparative-500_all.question"
    seed_tests = get_qca_data(input="./data/qca/qca_comparative-500.question")
    # seed_tests = seed_tests[:500]
    gen_dataset = random_gen_tests(seed_tests)
    save_qca_data(gen_dataset, out_path)


if __name__ == '__main__':
    # run_drqa(data_type="squad")
    # run_drqa(data_type="webquestions")
    # run_marl()
    run_albart("race")
