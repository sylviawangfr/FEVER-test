from utils.file_loader import read_json_rows, save_intermidiate_results
from sample_for_nli_esim.tf_idf_sample_v1_0 import convert_evidence2scoring_format
import utils.common_types as model_para
from BERT_test.ss_eval import ss_f1_score_and_save
import config
from utils import c_scorer

def read_upstream_files(bert_ss_data, gat_ss_data):
    pass


def update_score(bert_item, gat_item):
    pass


def get_gat_prob(gat_dict:dict, id, sent_id):
    if id not in gat_dict.keys():
        return 0
    gat_sents = gat_dict[id]["scored_sentids"]
    if len(gat_sents) < 1:
        return 0
    for i in gat_sents:
        if i[0] == sent_id:
            return i[2]
    return 0


def eval_and_save(bert_data, gat_data, output_folder, top_n=5, eval=True, save=True):
    merged = []
    bert_dict = dict()
    gat_dict = dict()
    for i in bert_data:
        bert_dict[i['id']] = i
    for j in gat_data:
        gat_dict[j['id']] = j
    for item in bert_data:
        scored_sentids_bert = item['scored_sentids']
        scored_sentids = []
        for c in scored_sentids_bert:
            sent_id = c[0]
            prob = c[2]
            gat_prob = get_gat_prob(gat_dict, item['id'], sent_id)
            new_prob = prob + gat_prob
            scored_sentids.append([sent_id, new_prob])
        scored_sentids = sorted(scored_sentids, key=lambda x: -x[1])
        predicted_sentids = [sid for sid, _ in scored_sentids][:top_n]
        predicted_evidence = convert_evidence2scoring_format(predicted_sentids)
        if eval:
            new_item = {"id": item["id"],
                        "verifiable": item["verifiable"],
                        "label": item['label'],
                        "claim": item['claim'],
                        "evidence": item['evidence'],
                        "scored_sentids": scored_sentids,
                        "predicted_sentids": predicted_sentids,
                        "predicted_evidence": predicted_evidence}
        else:
            new_item = {"id": item["id"],
                        "claim": item['claim'],
                        "scored_sentids": scored_sentids,
                        "predicted_sentids": predicted_sentids,
                        "predicted_evidence": predicted_evidence}
        merged.append(new_item)
    dev_original = read_json_rows(config.FEVER_DEV_JSONL)
    score_for_ss_evidence_list(merged, dev_original, output_folder, eval=eval, thresholds=0.1, top_n=top_n, save=save)


def score_for_ss_evidence_list(upstream_with_ss_evidence, original_data, output_dir, eval=True, thresholds=0.1, top_n=5, save=False):
    paras = model_para.PipelineParas()
    paras.output_folder = output_dir
    paras.original_data = original_data
    paras.prob_thresholds = thresholds
    if eval:
        paras.mode = 'dev'
    else:
        paras.mode = 'test'

    if paras.mode == 'dev':
        eval_mode = {'check_sent_id_correct': True, 'standard': False}
        if not isinstance(top_n, list):
            top_n = [top_n]

        for n in top_n:
            print(f"max evidence number:", n)
            strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(upstream_with_ss_evidence,
                                                                        paras.original_data,
                                                                        max_evidence=n,
                                                                        mode=eval_mode,
                                                                        error_analysis_file=paras.get_f1_log_file(
                                                                            f'{paras.prob_thresholds}_{n}_ss'),
                                                                        verbose=False)
            tracking_score = strict_score
            print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
            print("Strict score:", strict_score)
            print(f"Eval Tracking score:", f"{tracking_score}")
        if save:
            save_intermidiate_results(upstream_with_ss_evidence, paras.get_eval_data_file(f'bert_gat_ss_{n}'))
            print(f"results saved at: {paras.output_folder}")



if __name__ == '__main__':
    bert_data = read_json_rows(config.RESULT_PATH / "bert_ss_dev_10/eval_data_ss_10_dev_0.1_top[10].jsonl")
    gat_data = read_json_rows(config.RESULT_PATH / "gat_ss_dev_10/eval_data_ss_10_dev_0.1_top[10].jsonl")
    print("eval bert + gat dev result")
    eval_and_save(bert_data, gat_data, 'bert_gat_merged_ss_dev_5', save=False)

    print('-------------------------------\n')
    print("eval bert dev result")
    score_for_ss_evidence_list(bert_data, bert_data, 'bert_ss_dev_10', top_n=[10, 5])

    print('-------------------------------\n')
    print("eval gat dev result")
    score_for_ss_evidence_list(gat_data, gat_data, 'gat_ss_dev_10', top_n=[10, 5])

    print('-------------------------------\n')
    bert_data = read_json_rows(config.RESULT_PATH / "bert_ss_test_10/eval_data_ss_10_test_0.1_top[10].jsonl")
    gat_data = read_json_rows(config.RESULT_PATH / "gat_ss_test_10/eval_data_ss_10_test_0.1_top[10].jsonl")
    print("eval bert + gat test result, calculating")
    eval_and_save(bert_data, gat_data, 'bert_gat_merged_ss_test_5', eval=False, save=True)
    print("done")





