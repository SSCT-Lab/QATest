import json
import os
from collections import Counter


def get_webquestions_data(data_path):  # 获取原始数据集，从 DrQA的 datasets中读取 txt文件，返回初始种子的字典
    questions, answers, seed_tests = [], [], []
    f = open(data_path, 'r')
    data_set = f.readlines()
    for data in data_set:
        test_case = json.loads(data)
        questions.append(test_case['question'])
        answers.append(test_case['answer'])
    for q, a in zip(questions, answers):
        d = {'init_q': q, 'question': q, 'answer': a, 'is_init': True, 'aug_times': 0, 'aug': 'None', 'iter_times': 0}
        seed_tests.append(d)
    return seed_tests


def get_webquestions_gen_data(data_path):
    tests = []
    f = open(data_path, 'r')
    data_set = f.readlines()
    for data in data_set:
        test_case = json.loads(data)
        init_q = test_case['init_q']
        question = test_case['question']
        answer = test_case['answer']
        is_init = test_case['is_init']
        aug_times = test_case['aug_times']
        aug = test_case['aug']
        iter_times = test_case['iter_times']
        d = {'init_q': init_q, 'question': question, 'answer': answer, 'is_init': is_init, 'aug_times': aug_times,
             'aug': aug, 'iter_times': iter_times}
        tests.append(d)
    return tests


def separate_by_mr(file_path, out_dir):
    tests_mr1, tests_mr2, tests_mr3 = [], [], []
    f = open(file_path, 'r')
    data_set = f.readlines()
    for data in data_set:
        test_case = json.loads(data)
        aug = test_case['aug']
        if aug in ['back_translate', 'adverbial_preposition', 'insert_word']:
            tests_mr1.append(test_case)
        elif aug in ['synonym_replace', 'entity_replace', 'wps']:
            tests_mr2.append(test_case)
        elif aug in ['keybord_mistake', 'ocr_mistake', 'spelling_mistake', 'double_question_mark']:
            tests_mr3.append(test_case)
    save_webquestions_data(tests_mr1, os.path.join(out_dir, "webquestion_mr1.txt"))
    save_webquestions_data(tests_mr2, os.path.join(out_dir, "webquestion_mr2.txt"))
    save_webquestions_data(tests_mr3, os.path.join(out_dir, "webquestion_mr3.txt"))


def save_webquestions_data(tests, file_path):  # 将扩增后的数据整理成 txt，用于返回给 DrQA 预测
    with open(file_path, 'w') as f:  # 追加，不断更新测试集的大小
        for data in tests:
            f.write(json.dumps(data))
            f.write('\n')


def analysis_webquestions(data_path, bug_id_path):
    with open(bug_id_path, 'r', encoding='utf-8') as f:
        bug_id = f.readlines()
    tests = get_webquestions_gen_data(data_path)
    bug_tets_initq, bug_tets_iters, tets_iters, test_initq = [], [], [], []
    for item in tests:
        tets_iters.append(item['iter_times'])
        test_initq.append(item['init_q'])
    for i in bug_id:
        bug_tets_initq.append(tests[int(i)]['init_q'])
        bug_tets_iters.append(tests[int(i)]['iter_times'])
    print(len(dict(Counter(test_initq))))
    # print(dict(Counter(test_initq)))
    print("Num of seed tests to generate bug case:", len(dict(Counter(bug_tets_initq))))  # 有多少个种子数据产生出了至少一个bug用例
    print("Iter times and # bug cases:", dict(Counter(bug_tets_iters)))  # 变异次数对应的bug用例个数
    print("Iter times and # test cases:", dict(Counter(tets_iters)))  # 变异次数对应的总测试用例个数


if __name__ == '__main__':
    # separate_by_mr("../data/WebQuestions/WebQuestions-test-500_all.txt", "../data/WebQuestions/")
    analysis_webquestions("../data/WebQuestions/test/WebQuestions-500_aug_qatest.txt", "../data/WebQuestions/test/WebQuestions-500_aug_qatest_bug_id.txt")