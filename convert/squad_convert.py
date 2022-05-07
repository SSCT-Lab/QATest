import json
import os.path
from collections import defaultdict
import lib
from collections import Counter


def get_squad_data(input):
    # Read dataset
    with open(input, encoding='utf-8') as f:
        dataset = json.load(f)

    seed_tests = []
    # Iterate and write question-answer pairs
    for article in dataset['data']:
        title = article['title']
        for paragraph in article['paragraphs']:  # 每个paragraph底下对应一段context和多个QA对
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers']
                is_impossible = qa['is_impossible']
                id = qa['id']
                if is_impossible == "true":
                    plausible_answers = qa['plausible_answers']
                    d = {'init_q': question, 'question': question, 'answers': answer, 'is_init': True, 'aug_times': 0,
                         'aug': 'None', 'is_impossible': is_impossible, 'plausible_answers': plausible_answers,
                         'context': context, 'title': title, 'id': id, 'iter_times': 0}
                else:
                    d = {'init_q': question, 'question': question, 'answers': answer, 'is_init': True, 'aug_times': 0,
                         'aug': 'None', 'is_impossible': is_impossible, 'context': context, 'title': title, 'id': id,
                         'iter_times': 0}
                seed_tests.append(d)
    return seed_tests


def get_squad_gen_data(input):
    # Read dataset
    with open(input, encoding='utf-8') as f:
        dataset = json.load(f)

    tests = []
    # Iterate and write question-answer pairs
    for article in dataset['data']:
        title = article['title']
        for paragraph in article['paragraphs']:  # 每个paragraph底下对应一段context和多个QA对
            context = paragraph['context']
            for qa in paragraph['qas']:
                init_q = qa['init_q']
                question = qa['question']
                aug_times = qa['aug_times']
                aug = qa['aug']
                answer = qa['answers']
                is_impossible = qa['is_impossible']
                id = qa['id']
                iter_times = qa['iter_times']
                if is_impossible == "true":
                    plausible_answers = qa['plausible_answers']
                    d = {'init_q': init_q, 'question': question, 'answers': answer, 'aug_times': aug_times,
                         'aug': aug, 'is_impossible': is_impossible, 'plausible_answers': plausible_answers,
                         'context': context, 'title': title, 'id': id, 'iter_times': iter_times}
                else:
                    d = {'init_q': init_q, 'question': question, 'answers': answer, 'aug_times': aug_times,
                         'aug': aug, 'is_impossible': is_impossible, 'context': context, 'title': title, 'id': id,
                         'iter_times': iter_times}
                tests.append(d)
    return tests


def save_squad_data(tests, file_path):
    save_dic = dict()
    save_dic['version'] = 'aug1'
    save_dic['data'] = []

    rows_by_title = defaultdict(list)
    for data in tests:
        rows_by_title[data['title']].append(data)

    for title, par_li in rows_by_title.items():
        # print("title:", title)
        paragraphs_li = []
        rows_by_context = defaultdict(list)
        for d in par_li:
            rows_by_context[d['context']].append(d)
        for context, qas_li in rows_by_context.items():
            # print("context:", context)
            qua_li = []
            for qas in qas_li:
                question = qas['question']
                answers = qas['answers']
                is_impossible = qas['is_impossible']
                init_q = qas['init_q']
                aug_times = qas['aug_times']
                iter_times = qas['iter_times']
                aug = qas['aug']
                id = lib.generate_random_str(24)
                if is_impossible == "true":
                    plausible_answers = qas['plausible_answers']
                    tmp = {'question': question, 'id': id, 'answers': answers, 'is_impossible': is_impossible,
                           'plausible_answers': plausible_answers, 'init_q': init_q, 'aug_times': aug_times, 'aug': aug,
                           'iter_times': iter_times}
                else:
                    tmp = {'question': question, 'id': id, 'answers': answers, 'is_impossible': is_impossible,
                           'init_q': init_q, 'aug_times': aug_times, 'aug': aug, 'iter_times': iter_times}
                qua_li.append(tmp)
            paragraphs_li.append({'qas': qua_li, 'context': context})
        save_dic['data'].append({'title': title, 'paragraphs': paragraphs_li})
    # print(save_dic)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(save_dic, ensure_ascii=False, indent=2))


def separate_by_mr(file_path, out_dir):
    # Read dataset
    with open(file_path, encoding='utf-8') as f:
        dataset = json.load(f)

    tests_mr1, tests_mr2, tests_mr3 = [], [], []
    # Iterate and write question-answer pairs
    for article in dataset['data']:
        title = article['title']
        for paragraph in article['paragraphs']:  # 每个paragraph底下对应一段context和多个QA对
            context = paragraph['context']
            for qa in paragraph['qas']:
                aug = qa['aug']
                if qa['is_impossible'] == "true":
                    plausible_answers = qa['plausible_answers']
                    tmp = {'question': qa['question'], 'id': qa['id'], 'answers': qa['answers'], 'aug': aug,
                           'is_impossible': qa['is_impossible'], 'plausible_answers': plausible_answers,
                           'init_q': qa['init_q'], 'aug_times': qa['aug_times'], 'iter_times': qa['iter_times'],
                           'context': context, 'title': title}
                else:
                    tmp = {'question': qa['question'], 'id': qa['id'], 'answers': qa['answers'],
                           'is_impossible': qa['is_impossible'], 'init_q': qa['init_q'], 'aug_times': qa['aug_times'],
                           'aug': aug, 'iter_times': qa['iter_times'], 'context': context, 'title': title}
                if aug in ['back_translate', 'adverbial_preposition', 'insert_word']:
                    tests_mr1.append(tmp)
                elif aug in ['synonym_replace', 'entity_replace', 'wps']:
                    tests_mr2.append(tmp)
                elif aug in ['keybord_mistake', 'ocr_mistake', 'spelling_mistake', 'double_question_mark']:
                    tests_mr3.append(tmp)
    mr1_path = os.path.join(out_dir, "squad_mr1.json")
    mr2_path = os.path.join(out_dir, "squad_mr2.json")
    mr3_path = os.path.join(out_dir, "squad_mr3.json")
    save_squad_data(tests_mr1, mr1_path)
    save_squad_data(tests_mr2, mr2_path)
    save_squad_data(tests_mr3, mr3_path)


def analysis_squad(data_path, bug_id_path):
    with open(bug_id_path, 'r', encoding='utf-8') as f:
        bug_id = f.readlines()
    tests = get_squad_gen_data(data_path)
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
    # analysis_squad("../data/squad/test/squad-dev-v2.0-500_aug_qatest.json", "../data/squad/test/albert_qatest_bug_case.txt")
    seed = get_squad_data("../data/squad/squad-dev-v2.0.json")
    print(len(seed))
