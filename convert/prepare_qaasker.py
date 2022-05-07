import json
import csv


def convert_squad2(input_json_path, output_tsv_path):
    with open(input_json_path, encoding='utf-8') as f:
        dataset = json.load(f)

    seed_tests = []
    # Iterate and write question-answer pairs
    for article in dataset['data']:
        title = article['title']
        for paragraph in article['paragraphs']:  # 每个paragraph底下对应一段context和多个QA对
            context = paragraph['context'].replace('\n', '').replace('\r', '')
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers']
                is_impossible = qa['is_impossible']
                id = qa['id']
                if is_impossible == "true":
                    plausible_answers = qa['plausible_answers']
                    d = {'init_q': question, 'question': question, 'answer': answer, 'is_init': True, 'aug_times': 0,
                         'aug': 'None', 'is_impossible': is_impossible, 'plausible_answers': plausible_answers,
                         'context': context, 'title': title, 'id': id, 'iter_times': 0}
                else:
                    d = {'init_q': question, 'question': question, 'answer': answer, 'is_init': True, 'aug_times': 0,
                         'aug': 'None', 'is_impossible': is_impossible, 'context': context, 'title': title, 'id': id,
                         'iter_times': 0}
                seed_tests.append(d)

    with open(output_tsv_path, 'w', newline='', encoding='utf-8') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for data in seed_tests:
            tmp_context = "(" + data['title'] + ")" + data['context']
            question_and_paragraph = " \\n ".join([data['question'], tmp_context])
            answer = data['answer']
            tmp_ans = []
            if len(answer) == 0:
                tmp_ans.append('<No Answer>')
            else:
                for a in answer:
                    tmp_ans.append(a['text'])
            tsv_w.writerow([question_and_paragraph, tmp_ans])  # 单行写入


def convert_squad1(input_json_path, output_tsv_path):
    with open(input_json_path, encoding='utf-8') as f:
        dataset = json.load(f)

    seed_tests = []
    # Iterate and write question-answer pairs
    for article in dataset['data']:
        title = article['title']
        for paragraph in article['paragraphs']:  # 每个paragraph底下对应一段context和多个QA对
            context = paragraph['context'].replace('\n', '').replace('\r', '')
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers']
                id = qa['id']
                d = {'init_q': question, 'question': question, 'answer': answer, 'is_init': True, 'aug_times': 0,
                     'aug': 'None', 'context': context, 'title': title, 'id': id}
                seed_tests.append(d)

    with open(output_tsv_path, 'w', newline='', encoding='utf-8') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for data in seed_tests:
            tmp_context = "(" + data['title'] + ")" + data['context']
            question_and_paragraph = " \\n ".join([data['question'], tmp_context])
            answer = data['answer']
            tmp_ans = []
            if len(answer) == 0:
                tmp_ans.append('<No Answer>')
            else:
                for a in answer:
                    tmp_ans.append(a['text'])
            tsv_w.writerow([question_and_paragraph, str(tmp_ans)])  # 单行写入


def convert_webquestions(input_txt_path, output_tsv_path):
    questions, answers = [], []
    f = open(input_txt_path, 'r')
    data_set = f.readlines()
    for data in data_set:
        test_case = json.loads(data)
        questions.append(test_case['question'])
        answers.append(test_case['answer'])

    with open(output_tsv_path, 'w', newline='', encoding='utf-8') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for question, answer in zip(questions, answers):
            tsv_w.writerow([question, answer])


def get_predictions(pre_path, pre_tsv_path):
    f = open(pre_path, 'r')
    data_set = f.readlines()
    preds = []
    for data in data_set:
        dict_data = eval(data)[0]
        pred = dict_data['span']
        preds.append(pred)
    with open(pre_tsv_path, 'w', newline='', encoding='utf-8') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for pred in preds:
            tsv_w.writerow([pred])


if __name__ == '__main__':
    # convert_squad2("../data/squad/squad-dev-v2.0-500.json", "squad-dev-v2.0-500.tsv")
    # convert_squad1("../data/squad1.1/SQuAD-v1.1-dev-500.json", "SQuAD-v1.1-dev-500.tsv")
    # convert_webquestions("../data/WebQuestions/WebQuestions-test-500.txt", "WebQuestions-test-500.tsv")
    get_predictions("SQuAD-v1.1-dev-500-default-pipeline.preds", "SQuAD-v1.1-dev-500_out.tsv")
    get_predictions("WebQuestions-test-500-default-pipeline.preds", "WebQuestions-test-500_out.tsv")
