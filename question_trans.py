import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import tagme
import time
import requests

# 标注的“Authorization Token”，需要注册才有
tagme.GCUBE_TOKEN = "7ecd57b0-af37-4f2f-ba27-07721d933155-843339462"


#  1. 键盘输入导致的错误
def keybord_mistake(question, char_max=1, word_max=1, special_char=False, numeric=False, upper_case=False):
    aug = nac.KeyboardAug(aug_char_max=char_max, aug_word_max=word_max, include_special_char=special_char,
                          include_numeric=numeric, include_upper_case=upper_case)
    trans_question = aug.augment(question)
    return trans_question


#  2. OCR识别导致的错误
def ocr_mistake(question, char_max=1, word_max=1):
    aug = nac.OcrAug(aug_char_max=char_max, aug_word_max=word_max)
    trans_question = aug.augment(question)
    return trans_question


#  3. 单词拼写错误
def spelling_mistake(question, aug_max=1):
    aug = naw.SpellingAug(aug_max=aug_max)
    trans_question = aug.augment(question)
    return trans_question


#  4. 同义词替换，基于 wordnet
def synonym_replace(question, aug_max=1):
    aug = naw.SynonymAug(aug_max=aug_max)
    trans_question = aug.augment(question)
    return trans_question


# 5. 状语提前，when引导的时间状语、if引导的条件状语
def adverbial_preposition(question):
    question = question.replace('\n', '').replace('\r', '')
    if question.lower().find(" if ") > 0:
        above, below = question.lower().split(" if ", 1)
        # print(above, "\n", below)
        trans_question = "If " + below[:-1] + ", " + above[0].lower() + above[1:] + "?"
        # print(trans_question)
        return trans_question
    elif question.lower().find(" when ") > 0:
        above, below = question.lower().split(" when ", 1)
        # print(above, "\n", below)
        trans_question = "When " + below[:-1] + ", " + above[0].lower() + above[1:] + "?"
        # print(trans_question)
        return trans_question
    else:
        return question


# 6. 单词插入，基于预训练 bert-base-uncased模型
def insert_word(question):
    aug = naw.ContextualWordEmbsAug(action="insert", model_path='D:/pre_trained_models/bert-base-uncased', device='cuda')
    trans_question = aug.augment(question)
    return trans_question


# 7. 反转翻译，基于facebook/wmt19-en-de和facebook/wmt19-de-en，英文德文互转
def back_translate(question):
    aug = naw.BackTranslationAug(from_model_name='D:/pre_trained_models/facebook/wmt19-en-de',
                                 to_model_name='D:/pre_trained_models/facebook/wmt19-de-en')
    trans_question = aug.augment(question)
    return trans_question


# 8. 实体识别后进行 wiki 实体映射替换
def entity_replace(question):
    theta = 0.1
    try:
        annotations = tagme.annotate(question, lang="en")
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
        # raise Exception(e)  # 其他异常，抛出来
        return question  # 直接返回原始句子

    annotate_dic = dict()
    if annotations is None or annotations.get_annotations(theta) is None:
        return question

    else:
        for ann in annotations.get_annotations(theta):
            A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], \
                          str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
            annotate_dic[(A, B)] = score

    if len(annotate_dic) == 0:
        return question
    else:
        dic = sorted(annotate_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # 按照key降序排序，key越高，置信度越高
        tmp_dic = dic[0]  # 得分最高的一对 (('what country', 'What a Country!'), '0.4545454680919647')
        ori = tmp_dic[0][0]
        new = tmp_dic[0][1]
        score = tmp_dic[1]
        # print("ori:", ori, ", new:", new, ", score:", score)
        if ori.lower() is not new.lower():
            trans_question = question.replace(ori, new)
            return trans_question
        else:
            return question


# 9. WP [Wh-pronoun 代词（who whose which）] → WP's
def wps(question):
    if question.startswith('What is'):
        trans_question = question.replace("What is", "What's")
    elif question.startswith("Who is"):
        trans_question = question.replace("Who is", "Who's")
    elif question.startswith("Where is"):
        trans_question = question.replace("Where is", "Where's")
    elif question.startswith("When is"):
        trans_question = question.replace("When is", "When's")
    elif question.startswith("How is"):
        trans_question = question.replace("How is", "How's")
    else:
        trans_question = question
    return trans_question


# 10. ? → ??
def double_question_mark(question):
    if question.endswith("?") and question.count("?") == 1:  # 有且只有一个问号在句子最末尾
        trans_question = question + "?"
    elif not question.endswith("?"):  # 末尾没有问号
        trans_question = question + "??"
    else:
        trans_question = question
    return trans_question


if __name__ == '__main__':
    # questions = ["When was the Latin version of the word Norman first recorded?",
    #              "When was the French version of the word Norman first recorded??"]
    #
    # for question in questions:
    #     print("1. keybord_mistake:", keybord_mistake(question))
    #     print("2. ocr_mistake:", ocr_mistake(question))
    #     print("3. spelling_mistake:", spelling_mistake(question))
    #     print("4. synonym_replace:", synonym_replace(question))
    #     print("5. adverbial_preposition:", adverbial_preposition(question))
    #     print("6. insert_word:", insert_word(question))
    #     print("7. back_translate:", back_translate(question))
    #     print("8. entity_replace:", entity_replace(question))
    #     print("9. wps:", wps(question))
    #     print("10. double_question_mark:", double_question_mark(question))
    print(synonym_replace("What is a string over a Greek number when considering a computational problem?"))
