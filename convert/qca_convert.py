import json
import lib
import os
from collections import Counter


def get_qca_data(input):
    seed_tests = []
    with open(input, encoding='utf-8') as f:
        dataset = json.load(f)

    for item in dataset:
        title = item
        split = title.rfind(')')
        question_type = title[0:split+1]
        id = title[split+1:]
        question = dataset[item]['question']
        entity = dataset[item]['entity']
        relation = dataset[item]['relation']
        type = dataset[item]['type']
        response_entities = dataset[item]['response_entities']
        orig_response = dataset[item]['orig_response']
        entity_mask = dataset[item]['entity_mask']
        relation_mask = dataset[item]['relation_mask']
        type_mask = dataset[item]['type_mask']
        input = dataset[item]['input']

        d = {'init_q': question, 'question': question, 'id': id, 'entity': entity, 'relation': relation, 'type': type,
             'response_entities': response_entities, 'orig_response': orig_response, 'entity_mask': entity_mask,
             'relation_mask': relation_mask, 'type_mask': type_mask, 'input': input, 'question_type': question_type,
             'is_init': True, 'aug_times': 0, 'aug': 'None', 'iter_times': 0}
        seed_tests.append(d)
    return seed_tests


def get_qca_gen_data(input):
    seed_tests = []
    with open(input, encoding='utf-8') as f:
        dataset = json.load(f)

    for item in dataset:
        title = item
        split = title.rfind(')')
        question_type = title[0:split + 1]
        id = title[split + 1:]
        question = dataset[item]['question']
        init_q = dataset[item]['init_q']
        entity = dataset[item]['entity']
        relation = dataset[item]['relation']
        type = dataset[item]['type']
        response_entities = dataset[item]['response_entities']
        orig_response = dataset[item]['orig_response']
        entity_mask = dataset[item]['entity_mask']
        relation_mask = dataset[item]['relation_mask']
        type_mask = dataset[item]['type_mask']
        input = dataset[item]['input']
        is_init = dataset[item]['is_init']
        aug_times = dataset[item]['aug_times']
        aug = dataset[item]['aug']
        iter_times = dataset[item]['iter_times']

        d = {'init_q': init_q, 'question': question, 'id': id, 'entity': entity, 'relation': relation, 'type': type,
             'response_entities': response_entities, 'orig_response': orig_response, 'entity_mask': entity_mask,
             'relation_mask': relation_mask, 'type_mask': type_mask, 'input': input, 'question_type': question_type,
             'is_init': is_init, 'aug_times': aug_times, 'aug': aug, 'iter_times': iter_times}
        seed_tests.append(d)
    return seed_tests


def save_qca_data(tests, file_path):
    save_dic = dict()
    for data in tests:
        question_type = data['question_type']
        id = lib.generate_random_num(8)
        item = question_type + id
        save_dic[item] = {'question': data['question'], 'entity': data['entity'], 'relation': data['relation'],
                          'type': data['type'], 'response_entities': data['response_entities'],
                          'orig_response': data['orig_response'], 'entity_mask': data['entity_mask'],
                          'relation_mask': data['relation_mask'], 'type_mask': data['type_mask'],
                          'input': data['input'], 'init_q': data['init_q'], 'is_init': data['is_init'],
                          'aug_times': data['aug_times'], 'aug': data['aug'], 'iter_times': data['iter_times']}
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(save_dic, ensure_ascii=False, indent=2))


def separate_by_mr(file_path, out_dir):
    tests_mr1 = dict()
    tests_mr2 = dict()
    tests_mr3 = dict()
    with open(file_path, encoding='utf-8') as f:
        dataset = json.load(f)
    for item in dataset:
        if dataset[item]['aug'] in ['back_translate', 'adverbial_preposition', 'insert_word']:
            tests_mr1[item] = dataset[item]
        elif dataset[item]['aug'] in ['synonym_replace', 'entity_replace', 'wps']:
            tests_mr2[item] = dataset[item]
        elif dataset[item]['aug'] in ['keybord_mistake', 'ocr_mistake', 'spelling_mistake', 'double_question_mark']:
            tests_mr3[item] = dataset[item]
    with open(os.path.join(out_dir, "qca_verification_mr1.question"), 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(tests_mr1, ensure_ascii=False, indent=2))
    with open(os.path.join(out_dir, "qca_verification_mr2.question"), 'w', encoding='utf-8') as f2:
        f2.write(json.dumps(tests_mr2, ensure_ascii=False, indent=2))
    with open(os.path.join(out_dir, "qca_verification_mr3.question"), 'w', encoding='utf-8') as f3:
        f3.write(json.dumps(tests_mr3, ensure_ascii=False, indent=2))


def analysis_qca(data_path, bug_id_path):
    with open(bug_id_path, 'r', encoding='utf-8') as f:
        bug_id = f.readlines()
    tests = get_qca_gen_data(data_path)
    print("total tests:", len(tests))
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


if __name__ == '__main__':
    seed_tests = get_qca_data("../data/qca/qca_comparative.question")
    print(len(seed_tests))
    # seed_tests = seed_tests[:500]
    # save_qca_data(seed_tests, "../data/qca/qca_verification-500.question")
    # seed_tests = get_qca_data("../data/qca/test/qca_comparative-500_aug_qatest.question")
    # print(len(seed_tests))
    # separate_by_mr("../data/qca/qca_verification-500_all.question", "../data/qca/")
    # analysis_qca("../data/qca/qca_comparative_mr3.question", "../data/qca/qca_comparative_mr3_bug_id.txt")
