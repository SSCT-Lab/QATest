import csv
import json
from collections import Counter
import os


def get_boolq_data(input):
    seed_tests = []
    with open(input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        question = line['question']
        title = line['title']
        answer = line['answer']
        passage = line['passage']
        d = {'init_q': question, 'question': question, 'answer': answer, 'passage': passage, 'title': title,
             'is_init': True, 'aug_times': 0, 'aug': 'None', 'iter_times': 0}
        seed_tests.append(d)
    return seed_tests


def get_boolq_gen_data(input):
    tests = []
    with open(input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        question = line['question']
        init_q = line['init_q']
        title = line['title']
        answer = line['answer']
        passage = line['passage']
        is_init = line['is_init']
        aug_times = line['aug_times']
        aug = line['aug']
        iter_times = line['iter_times']
        d = {'init_q': init_q, 'question': question, 'answer': answer, 'passage': passage, 'title': title,
             'is_init': is_init, 'aug_times': aug_times, 'aug': aug, 'iter_times': iter_times}
        tests.append(d)
    return tests


def save_boolq_data(tests, tsv_file_path, jsonl_file_path):
    with open(tsv_file_path, 'w', newline='', encoding='utf-8') as f1, open(jsonl_file_path, 'w', newline='',
                                                                            encoding='utf-8') as f2:
        tsv_w = csv.writer(f1, delimiter='\t')
        for data in tests:
            question_and_paragraph = " \\n ".join([data['question'], data['passage']])
            answer = data['answer']
            tsv_w.writerow([question_and_paragraph, answer, data['is_init'], data['aug_times'], data['aug'],
                            data['iter_times']])  # 单行写入
            json.dump(data, f2)
            f2.write("\n")


def analysis_boolq(data_path, bug_id_path):
    with open(bug_id_path, 'r', encoding='utf-8') as f:
        bug_id = f.readlines()
    tests = get_boolq_gen_data(data_path)
    bug_tets_initq, bug_tets_iters, tets_iters, test_initq = [], [], [], []
    for item in tests:
        tets_iters.append(item['iter_times'])
        test_initq.append(item['init_q'])
    for i in bug_id:
        bug_tets_initq.append(tests[int(i)]['init_q'])
        bug_tets_iters.append(tests[int(i)]['iter_times'])
    print(len(dict(Counter(test_initq))))
    # print(dict(Counter(test_initq)))
    print("Num of seed tests to generate bug case:", len(set(bug_tets_initq)))  # 有多少个种子数据产生出了至少一个bug用例
    print("Iter times and # bug cases:", dict(Counter(bug_tets_iters)))  # 变异次数对应的bug用例个数
    print("Iter times and # test cases:", dict(Counter(tets_iters)))  # 变异次数对应的总测试用例个数


def separate_by_mr(file_path, out_dir):
    tests_mr1, tests_mr2, tests_mr3 = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        question = line['question']
        init_q = line['init_q']
        title = line['title']
        answer = line['answer']
        passage = line['passage']
        is_init = line['is_init']
        aug_times = line['aug_times']
        aug = line['aug']
        iter_times = line['iter_times']
        tmp = {'init_q': init_q, 'question': question, 'answer': answer, 'passage': passage, 'title': title,
             'is_init': is_init, 'aug_times': aug_times, 'aug': aug, 'iter_times': iter_times}
        if aug in ['back_translate', 'adverbial_preposition', 'insert_word']:
            tests_mr1.append(tmp)
        elif aug in ['synonym_replace', 'entity_replace', 'wps']:
            tests_mr2.append(tmp)
        elif aug in ['keybord_mistake', 'ocr_mistake', 'spelling_mistake', 'double_question_mark']:
            tests_mr3.append(tmp)
    save_boolq_data(tests_mr1, os.path.join(out_dir, "boolq_mr1.tsv"), os.path.join(out_dir, "boolq_mr1.jsonl"))
    save_boolq_data(tests_mr2, os.path.join(out_dir, "boolq_mr2.tsv"), os.path.join(out_dir, "boolq_mr2.jsonl"))
    save_boolq_data(tests_mr3, os.path.join(out_dir, "boolq_mr3.tsv"), os.path.join(out_dir, "boolq_mr3.jsonl"))


if __name__ == '__main__':
    # file_path = "../data/boolq/dev-500_all.jsonl"
    # separate_by_mr(file_path, "../data/boolq/")
    # analysis_boolq("../data/boolq/dev-1000_all.jsonl", "../data/boolq/dev-1000_all_bug_case.txt")
    analysis_boolq("../data/boolq/test/dev-500_aug_nocov.jsonl", "../data/boolq/test/dev-500_aug_nocov_bug_case.txt")
