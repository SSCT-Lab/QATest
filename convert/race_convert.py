import json
from collections import defaultdict
import os
from collections import Counter


def get_race_data(input):
    seed_tests = []
    with open(input, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            article = data['article']
            id = data['id']
            for i in range(len(data['answers'])):
                answer = data['answers'][i]
                option = data['options'][i]
                question = data['questions'][i]
                if '_' not in question and len(question) > 1:
                    d = {'init_q': question, 'question': question, 'answer': answer, 'option': option,
                         'article': article, 'is_init': True, 'aug_times': 0, 'aug': 'None', 'iter_times': 0, 'id': id}
                    seed_tests.append(d)
    return seed_tests


def get_race_gen_data(input):
    tests = []
    with open(input, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            article = data['article']
            id = data['id']
            for i in range(len(data['answers'])):
                answer = data['answers'][i]
                option = data['options'][i]
                question = data['questions'][i]
                init_q = data['init_qs'][i]
                is_init = data['is_init'][i]
                aug_times = data['aug_times'][i]
                aug = data['aug'][i]
                iter_times = data['iter_times'][i]

                d = {'init_q': init_q, 'question': question, 'answer': answer, 'option': option, 'aug': aug,
                     'article': article, 'is_init': is_init, 'aug_times': aug_times, 'iter_times': iter_times, 'id': id}
                tests.append(d)
    return tests


def save_race_data(tests, file_path):
    rows_by_id = defaultdict(list)
    for data in tests:
        rows_by_id[data['id']].append(data)

    save = []
    for id, item in rows_by_id.items():
        answers, options, questions, is_inits, aug_times, augs, iter_times, init_qs = [], [], [], [], [], [], [], []
        article = item[0]['article']
        for i in range(len(item)):
            answers.append(item[i]['answer'])
            options.append(item[i]['option'])
            questions.append(item[i]['question'])
            is_inits.append(item[i]['is_init'])
            aug_times.append(item[i]['aug_times'])
            augs.append(item[i]['aug'])
            iter_times.append(item[i]['iter_times'])
            init_qs.append(item[i]['init_q'])

        json_dic = {'answers': answers, 'options': options, 'questions': questions, 'article': article, 'id': id,
                    'is_init': is_inits, 'aug_times': aug_times, 'aug': augs, 'iter_times': iter_times, 'init_qs': init_qs}
        save.append(json_dic)
        # print(json_dic)

    with open(file_path, 'w', encoding='utf-8') as f:
        for jsonl in save:
            json.dump(jsonl, f)
            f.write("\n")


def separate_by_mr(file_path, out_dir):
    tests_mr1, tests_mr2, tests_mr3 = [], [], []
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            for i in range(len(data['aug'])):
                aug = data['aug'][i]
                tmp = {'init_q': data['init_qs'][i], 'question': data['questions'][i], 'answer': data['answers'][i],
                       'option': data['options'][i], 'article': data['article'], 'is_init': data['is_init'][i],
                       'aug_times': data['aug_times'][i], 'aug': data['aug'][i], 'iter_times': data['iter_times'][i],
                       'id': data['id']}
                if aug in ['back_translate', 'adverbial_preposition', 'insert_word']:
                    tests_mr1.append(tmp)
                elif aug in ['synonym_replace', 'entity_replace', 'wps']:
                    tests_mr2.append(tmp)
                elif aug in ['keybord_mistake', 'ocr_mistake', 'spelling_mistake', 'double_question_mark']:
                    tests_mr3.append(tmp)
    mr1_path = os.path.join(out_dir, "race_mr1.txt")
    mr2_path = os.path.join(out_dir, "race_mr2.txt")
    mr3_path = os.path.join(out_dir, "race_mr3.txt")
    save_race_data(tests_mr1, mr1_path)
    save_race_data(tests_mr2, mr2_path)
    save_race_data(tests_mr3, mr3_path)


def analysis_race(data_path, bug_id_path):
    with open(bug_id_path, 'r', encoding='utf-8') as f:
        bug_id = f.readlines()
    tests = get_race_gen_data(data_path)
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
    seed = get_race_data("../data/race/race_test_high.txt")
    print(len(seed))
    # save_race_data(seed[:500], "../data/race/race_test_high-500.txt")
    # separate_by_mr("../data/race/race_test_high-500_all.txt", "../data/race/")
    # analysis_race("../data/race/test/race_test_high-500_aug_gram.txt", "../data/race/test/predict_results_gram_bug_id.txt")
