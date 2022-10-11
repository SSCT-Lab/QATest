import argparse
from convert.squad_convert import *
from convert.squad1_convert import *
from convert.race_convert import *
from convert.qca_convert import *
from convert.boolq_convert import *
#from convert.webquestions_convert import *


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Generating datasets for testing QA systems.")
    parse.add_argument('--dataset', required=True, choices=['boolq', 'race', 'squad', 'WebQuestions',
                                                            'qca-verification', 'qca-comparative'])
    parse.add_argument('--system', required=True, choices=['unifiedqa', 'albert', 'drqa', 'marl'])
    args = parse.parse_args()
    dataset = args.dataset
    system = args.system

    if dataset == "race" and system == "albert":
        path = "./data/race/race_test_high.txt"
        seed = get_race_data(path)
        seed = seed[:500]
        save_race_data(seed, "./data/race/race_test_high-500.txt")
    elif dataset == "squad" and (system == "albert" or system == "unifiedqa"):
        path = "./data/squad/squad-dev-v2.0.json"
        seed = get_squad_data(path)
        seed = seed[:500]
        save_squad_data(seed, "./data/squad/squad-dev-v2.0-500.json")
    elif dataset == "squad" and system == "drqa":
        path = "./data/squad1.1/SQuAD-v1.1-dev.json"
        seed = get_squad1_1_data(path)
        seed = seed[:500]
        save_squad1_1_data(seed, "./data/squad1.1/SQuAD-v1.1-dev-500.json")
    elif dataset == "qca-verification" and system == "marl":
        path = "./data/qca/qca_verification.question"
        seed = get_qca_data(path)
        seed = seed[:500]
        save_qca_data(seed, "./data/qca/qca_verification-500.question")
    elif dataset == "qca-comparative" and system == "marl":
        path = "./data/qca/qca_comparative.question"
        seed = get_qca_data(path)
        seed = seed[:500]
        save_qca_data(seed, "./data/qca/qca_comparative-500.question")