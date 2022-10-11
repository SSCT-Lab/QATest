import argparse
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
#from convert.webquestions_convert import *
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


def get_aug_weight(test_results):  # test_results 应该是一个字典，每个变异算子对应的bug次数统计
    left_key = list({'keybord_mistake', 'ocr_mistake', 'spelling_mistake', 'synonym_replace', 'adverbial_preposition',
                     'insert_word', 'back_translate', 'entity_replace', 'wps', 'double_question_mark'} - set(test_results.keys()))
    # print(left_key)
    if len(left_key) != 0:
        for item in left_key:
            test_results[item] = 0
    weights = [test_results['keybord_mistake'], test_results['ocr_mistake'], test_results['spelling_mistake'],
               test_results['synonym_replace'], test_results['adverbial_preposition'], test_results['insert_word'],
               test_results['back_translate'], test_results['entity_replace'], test_results['wps'],
               test_results['double_question_mark']]
    return weights


def get_seed_dict(seed_tests):
    seed_dict = {}
    for item in seed_tests:
        seed_dict[item['init_q']] = 1
    # print(seed_dict)
    return seed_dict


def pick_seed_batch(seed, seed_dict, batch_size=5):  # 输入的是一个list，里面每个元素都是一个字典，对应一个 Q-A pair
    p_li = []
    for item in seed:
        p_li.append(1/seed_dict[item['init_q']])
    p_normalization = [i/sum(p_li) for i in p_li]
    picked = np.random.choice(seed, batch_size, p=p_normalization, replace=False)
    return picked
# def pick_seed_batch(seed, batch_size=5):  # 输入的是一个list，里面每个元素都是一个字典，对应一个 Q-A pair
#     p_li = []
#     for item in seed:
#         p_li.append(1/(1 + item['aug_times'] + item['iter_times']))
#         # p_li.append(1-item['aug_times']/20)
#     p_normalization = [i/sum(p_li) for i in p_li]
#     picked = np.random.choice(seed, batch_size, p=p_normalization, replace=False)
#     return picked


def random_generate_cases(batch_seed, seeds, aug_tests, aug_weights=None, aug_num=1):  # 生成了之后就评估质量，只保留质量合格的生成用例
    # aug_seeds = []
    aug_batch = []  # 只记录这一批次新生成的测试用例
    for data in batch_seed:
        add_to_seed = True
        for times in range(aug_num):
            to_select_aug_id = list(range(10))
            for i in range(10):
                flag = True
                aug_id = random.choices(to_select_aug_id, weights=aug_weights, k=1)[0]
                aug_tpye = aug_dict.get(aug_id)
                aug_question = eval(aug_dict.get(aug_id))(data['question'])
                if aug_question == data['question'] or len(aug_question) == 0:  # 扩增失败，生成的问题和原始问题完全一样
                    flag = False
                for seed in seeds:  # 这里还要判断 aug_seed 与其他相同 seed['init_q'] 的 ['question'] 是否完全重合，重合的话舍弃
                    if seed['init_q'] == data['init_q'] and rouge1_score(aug_question, seed['question'])['r'] == 1:
                        flag = False
                if flag:
                    rouge1_f = rouge1_score(aug_question, data['question'])['f']
                    if rouge1_f > 0.5:  # 符合质量要求，放入生成测试集
                        generate_data = {'question': aug_question, 'is_init': False, 'aug_times': 0, 'aug': aug_tpye,
                                         'iter_times': data['iter_times']+1}
                        left_key = list(set(data.keys()) - {'question', 'is_init', 'aug_times', 'aug', 'iter_times'})
                        for key in left_key:
                            generate_data[key] = data[key]
                        aug_tests.append(generate_data)
                        aug_batch.append(generate_data)
                        data['aug_times'] = data['aug_times'] + 1  # 原始句子的变异次数增加 1
                    break
            if i == 9:  # 试了10次，都扩增失败，就直接从种子集合中删除这个种子，结束对这个种子的扩增变换
                # seeds.remove(data)  # 不用删除了，因为刚开始就是删除了的
                add_to_seed = False
                break
        if add_to_seed:
            seeds.append(data)
        if not data['is_init']:
            aug_tests.append(data)
    return aug_tests, aug_batch, seeds


def back_to_seed(batch_seed, seed, seed_dict):  # 评估是否能作为新的种子数据，若覆盖率提升则保留加入种子集
    dtmc_matrix = get_dtmc_matrix(seed)
    perplexity_li, gram_cov_li = [], []
    for item in batch_seed:
        perplexity_li.append(get_sentence_perplexity(item, dtmc_matrix))
        gram_cov_li.append(gram_evaluate(item, seed))
    min_pro_id = np.argmin(perplexity_li)
    max_gram_id = np.argmax(gram_cov_li)
    # print("probability_li:", perplexity_li, "\n", "gram_cov_li:", gram_cov_li)
    if min_pro_id != max_gram_id:
        seed.append(batch_seed[min_pro_id])
        seed.append(batch_seed[max_gram_id])
        seed_dict[batch_seed[min_pro_id]['init_q']] = seed_dict[batch_seed[min_pro_id]['init_q']] + 1
        seed_dict[batch_seed[max_gram_id]['init_q']] = seed_dict[batch_seed[max_gram_id]['init_q']] + 1
    else:
        seed.append(batch_seed[min_pro_id])
        seed_dict[batch_seed[min_pro_id]['init_q']] = seed_dict[batch_seed[min_pro_id]['init_q']] + 1
    return seed


def back_to_seed_gram(batch_seed, seed, seed_dict):
    gram_cov_li = []
    for item in batch_seed:
        gram_cov_li.append(gram_evaluate(item, seed))
    max_gram_id = np.argmax(gram_cov_li)
    seed.append(batch_seed[max_gram_id])
    seed_dict[batch_seed[max_gram_id]['init_q']] = seed_dict[batch_seed[max_gram_id]['init_q']] + 1
    return seed


def back_to_seed_pro(batch_seed, seed, seed_dict):
    dtmc_matrix = get_dtmc_matrix(seed)
    perplexity_li = []
    for item in batch_seed:
        perplexity_li.append(get_sentence_perplexity(item, dtmc_matrix))
    min_pro_id = np.argmin(perplexity_li)
    seed.append(batch_seed[min_pro_id])
    seed_dict[batch_seed[min_pro_id]['init_q']] = seed_dict[batch_seed[min_pro_id]['init_q']] + 1
    return seed

# def back_to_seed_pro(batch_seed, seed, seed_dict):
#     dtmc_matrix = get_dtmc_matrix(seed)
#     perplexity_li = []
#     for item in batch_seed:
#         perplexity_li.append(get_sentence_perplexity(item, dtmc_matrix))
#     max_pro_id = np.argmax(perplexity_li)
#     seed.append(batch_seed[max_pro_id])
#     seed_dict[batch_seed[max_pro_id]['init_q']] = seed_dict[batch_seed[max_pro_id]['init_q']] + 1
#     return seed


def run(seed_tests, save_path, seed_dict, strategy, iter_N, aug_W=None):
    aug_tests = []
    aug_num = []

    for i in range(iter_N):  # 迭代 n 次
        batch_seed = pick_seed_batch(seed_tests, seed_dict, batch_size=5)  # 每一次选取 m 个数据作为一个批次
        for item in batch_seed:  # 先把选出来的种子从种子列表中删除，生成完之后再放回去
            seed_tests.remove(item)
            if not item['is_init']:  # 不是原始种子却在种子列表里，说明这个数据也在测试集合里
                aug_tests.remove(item)

        aug_tests, aug_batch, seed_tests = random_generate_cases(batch_seed, seed_tests, aug_tests, aug_weights=aug_W)
        print("seed_dict:", seed_dict)
        if strategy == "pro":
            if len(aug_batch) != 0:
                seed_tests = back_to_seed_pro(aug_batch, seed_tests, seed_dict)
        elif strategy == "gram":
            if len(aug_batch) != 0:
                seed_tests = back_to_seed_gram(aug_batch, seed_tests, seed_dict)
        elif strategy == "qatest":
            if len(aug_batch) != 0:
                seed_tests = back_to_seed(aug_batch, seed_tests, seed_dict)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print("iter times:", i, ", seed size:", len(seed_tests), ", test size:", len(aug_tests))

        if i > 0 and i % 100 == 0:
            aug_num.append(len(aug_tests))
    aug_num.append(len(aug_tests))

    with open(save_path, "w", encoding='utf=8') as f:
        for num in aug_num:
            f.write(str(num))
            f.write("\n")
    return aug_tests, seed_tests, aug_num


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Generating datasets for testing QA systems.")
    parse.add_argument('--dataset', required=True, choices=['boolq', 'race', 'squad', 'WebQuestions',
                                                           'qca-verification', 'qca-comparative'])
    parse.add_argument('--system', required=True, choices=['unifiedqa', 'albert', 'drqa', 'marl'])
    parse.add_argument('--strategy', required=True, choices=['gram', 'pro', 'qatest', 'nocov'])
    args = parse.parse_args()
    dataset = args.dataset
    system = args.system
    strategy = args.strategy

    if dataset == "boolq" and system == "unifiedqa":
        seed_tests = get_boolq_data(input="./data/boolq/dev-500.jsonl")
        save_path = "./data/boolq/test/dev-500_aug_num_" + strategy + ".txt"
        seed_dict = get_seed_dict(seed_tests)
        aug_tests, seed_tests, aug_num = run(seed_tests, save_path, seed_dict, strategy, iter_N=3000)
        save_boolq_data(aug_tests, "./data/boolq/test/dev-500_aug_" + strategy + ".tsv", "./data/boolq/test/dev-500_aug_" + strategy + ".jsonl")
        save_boolq_data(seed_tests, "./data/boolq/test/dev-500_seed_" + strategy + ".tsv", "./data/boolq/test/dev-500_seed_" + strategy + ".jsonl")
    elif dataset == "race" and system == "albert":
        seed_tests = get_race_data(input="./data/race/race_test_high-500.txt")
        save_path = "./data/race/test/race_test_high-500_aug_num_" + strategy + ".txt"
        # seed_tests = seed_tests[:500]
        seed_dict = get_seed_dict(seed_tests)
        aug_tests, seed_tests, aug_num = run(seed_tests, save_path, seed_dict, strategy, iter_N=3000)
        save_race_data(aug_tests, "./data/race/test/race_test_high-500_aug_" + strategy + ".txt")
        save_race_data(seed_tests, "./data/race/test/race_test_high-500_seed_" + strategy + ".txt")
    elif dataset == "squad" and (system == "albert" or system == "unifiedqa"):
        seed_tests = get_squad_data(input="./data/squad/squad-dev-v2.0-500.json")
        save_path = "./data/squad/test/squad-dev-v2.0-500_aug_num_" + strategy + ".txt"
        # seed_tests = seed_tests[:500]
        seed_dict = get_seed_dict(seed_tests)
        aug_tests, seed_tests, aug_num = run(seed_tests, save_path, seed_dict, strategy, iter_N=3000)
        save_squad_data(aug_tests, "./data/squad/test/squad-dev-v2.0-500_aug_" + strategy + ".json")
        save_squad_data(seed_tests, "./data/squad/test/squad-dev-v2.0-500_seed_" + strategy + ".json")
    elif dataset == "WebQuestions" and system == "drqa":
        seed_tests = get_webquestions_data("./data/WebQuestions/WebQuestions-test-500.txt")
        save_path = "./data/WebQuestions/test/WebQuestions_aug_num_" + strategy + ".txt"
        seed_tests = seed_tests[:500]
        seed_dict = get_seed_dict(seed_tests)
        aug_tests, seed_tests, aug_num = run(seed_tests, save_path, seed_dict, strategy, iter_N=3000)
        save_webquestions_data(aug_tests, "./data/WebQuestions/test/WebQuestions-500_aug_" + strategy + ".txt")
        save_webquestions_data(seed_tests, "./data/WebQuestions/test/WebQuestions-500_seed_" + strategy + ".txt")
    elif dataset == "squad" and system == "drqa":
        seed_tests = get_squad1_1_data(input="./data/squad1.1/SQuAD-v1.1-dev-500.json")
        save_path = "./data/squad1.1/test/SQuAD-v1.1-dev-500_aug_num_" + strategy + ".txt"
        # seed_tests = seed_tests[:500]
        seed_dict = get_seed_dict(seed_tests)
        aug_tests, seed_tests, aug_num = run(seed_tests, save_path, seed_dict, strategy, iter_N=3000)
        save_squad1_1_data(aug_tests, "./data/squad1.1/test/SQuAD-v1.1-dev-500_aug_" + strategy + ".json")
        save_squad1_1_data(seed_tests, "./data/squad1.1/test/SQuAD-v1.1-dev-500_seed_" + strategy + ".json")
    elif dataset == "qca-verification" and system == "marl":
        seed_tests = get_qca_data("./data/qca/qca_verification-500.question")
        save_path = "./data/qca/test/qca_verification-500_aug_num_" + strategy + ".txt"
        # seed_tests = seed_tests[:500]
        seed_dict = get_seed_dict(seed_tests)
        aug_tests, seed_tests, aug_num = run(seed_tests, save_path, seed_dict, strategy, iter_N=3000)
        save_qca_data(aug_tests, "./data/qca/test/qca_verification-500_aug_" + strategy + ".question")
        save_qca_data(seed_tests, "./data/qca/test/qca_verification-500_seed_" + strategy + ".question")
    elif dataset == "qca-comparative" and system == "marl":
        seed_tests = get_qca_data("./data/qca/qca_comparative-500.question")
        save_path = "./data/qca/test/qca_comparative-500_aug_num_" + strategy + ".txt"
        # seed_tests = seed_tests[:500]
        seed_dict = get_seed_dict(seed_tests)
        aug_tests, seed_tests, aug_num = run(seed_tests, save_path, seed_dict, strategy, iter_N=3000)
        save_qca_data(aug_tests, "./data/qca/test/qca_comparative-500_aug_" + strategy + ".question")
        save_qca_data(seed_tests, "./data/qca/test/qca_comparative-500_seed_" + strategy + ".question")
