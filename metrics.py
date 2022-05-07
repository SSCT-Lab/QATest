from rouge import Rouge


def rouge1_score(hyps, refs):
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps, refs)  # hyps是预测值，refs是参考值
    # print("rouge_score:", rouge_score)
    rouge1 = rouge_score[0]["rouge-1"]
    return rouge1


if __name__ == '__main__':
    t1 = "the cat was found under the bed?"
    t2 = "the cat was found under the bed?"
    rouge1 = rouge1_score(t1, t2)
    rouge1_f = rouge1['f']
    print(rouge1_f)