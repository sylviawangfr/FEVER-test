from utils.file_loader import read_json_rows, save_intermidiate_results
from sample_for_nli_esim.tf_idf_sample_v1_0 import convert_evidence2scoring_format
import utils.common_types as model_para
from BERT_test.ss_eval import ss_f1_score_and_save
import config
import tqdm
from utils import c_scorer
from doc_retriv.doc_retrieve import retri_doc_and_update_item
from BERT_test.bert_data_processor import *
from BERT_test.ss_eval import pred_ss_and_save
from dbpedia_sampler.dbpedia_ss_sampler import convert_to_graph_sampler
from graph_modules.gat_dbpedia_train2 import pred_prob


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


def eval_and_save(bert_data, gat_data, output_folder, top_n=5, thresholds=0.1, eval=True, save=True):
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
            if new_prob >= thresholds:
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
    score_for_ss_evidence_list(merged, dev_original, output_folder, eval=eval, thresholds=thresholds, top_n=[top_n], save=save)
    return

def redo_error_items(error_items, mode='dev'):
    output_dir = config.RESULT_PATH / "error_rerun"
    # doc retrive
    for j in error_items:
        retri_doc_and_update_item(j)

    # bert_ss
    paras = bert_para.PipelineParas()
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    paras.output_folder = "error_rerun/ss_bert"
    paras.mode = mode
    paras.pred = True
    paras.top_n = [10]
    paras.sample_n = 10
    paras.prob_thresholds = 0.01
    ss_bert = pred_ss_and_save(paras)

    # gat_ss
    dbpedia_output = output_dir / "ss_gat" / f"dbpedia_sample.jsonl"
    convert_to_graph_sampler(ss_bert,
                             dbpedia_output,
                             pred=True)
    print("done with dbpedia sampler")
    model_path = config.SAVED_MODELS_PATH / 'gat_ss_0.0001_epoch400_65.856_66.430'
    gat_output = config.RESULT_PATH / "error_rerun/ss_gat"
    ss_gat = pred_prob(model_path, error_items, ss_bert, gat_output, thredhold=0.1, pred=True, gpu=2, eval=True)

    # merge
    merged = eval_and_save(ss_bert, ss_gat, output_dir, top_n=5, thresholds=0.1, eval=True, save=True)


def get_empty_items(upstream_l):
    empty_items = []
    for i in tqdm(upstream_l):
        if len(i['predicted_sentids']) < 1:
            empty_items.append(i)
    for j in empty_items:
        retri_doc_and_update_item(j)
    save_intermidiate_results(empty_items, config.RESULT_PATH / 'empty_items.jsonl')


def get_error_items(upstream_l, output_dir, top_n=5):
    error_l = []
    for i in upstream_l:
        if not c_scorer.is_evidence_correct(i, top_n):
            error_l.append(i)
    save_intermidiate_results(error_l, output_dir / "error_ss.jsonl")


def score_for_ss_evidence_list(upstream_with_ss_evidence, original_data, output_dir, eval=True, thresholds=0.5, top_n=[5], save=False):
    paras = model_para.PipelineParas()
    paras.output_folder = output_dir
    paras.original_data = original_data
    paras.prob_thresholds = thresholds
    if eval:
        paras.mode = 'eval'
    else:
        paras.mode = 'test'

    if paras.mode == 'eval':
        eval_mode = {'check_sent_id_correct': True, 'standard': False}
        if not isinstance(top_n, list):
            top_n = [top_n]

        for n in top_n:
            print(f"max evidence number:", n)
            c_scorer.get_ss_recall_precision(upstream_with_ss_evidence, top_n=n, threshold=thresholds)
            strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(upstream_with_ss_evidence,
                                                                        paras.original_data,
                                                                        max_evidence=n,
                                                                        mode=eval_mode,
                                                                        error_analysis_file=paras.get_f1_log_file(
                                                                            f'{paras.prob_thresholds}_{n}_ss'),
                                                                        verbose=False)
            # tracking_score = strict_score
            print(f"Dev(strict/raw_acc/pr/rec/f1):{strict_score}/{acc_score}/{pr}/{rec}/{f1}/")
            # print("Strict score:", strict_score)
            # print(f"Eval Tracking score:", f"{tracking_score}")
        if save:
            save_intermidiate_results(upstream_with_ss_evidence, paras.get_eval_data_file(f'bert_gat_ss_{n}'))
            print(f"results saved at: {paras.output_folder}")
        return upstream_with_ss_evidence



if __name__ == '__main__':
    original_data = read_json_rows(config.FEVER_DEV_JSONL)
    bert_data = read_json_rows(config.RESULT_PATH / "bert_ss_dev_10/eval_data_ss_10_dev_0.1_top[10].jsonl")
    # get_error_items(bert_data, config.RESULT_PATH, top_n=5)
    gat_data = read_json_rows(config.RESULT_PATH / "gat_ss_dev_10/eval_data_ss_10_dev_0.1_top[10].jsonl")
    # print("eval bert + gat dev result")
    eval_and_save(bert_data, gat_data, 'bert_gat_merged_ss_dev_5', thresholds=0.1, top_n=5, save=False)
    # c_scorer.get_macro_ss_recall_precision(read_json_rows(config.RESULT_PATH / 'bert_gat_merged_ss_dev_5/eval_data_bert_gat_ss_5_dev_0.1_top[5].jsonl'))
    print('-------------------------------\n')
    print("eval bert dev result")
    score_for_ss_evidence_list(bert_data, original_data, 'bert_ss_dev_10', thresholds=0.4, top_n=[5])
    # c_scorer.get_macro_ss_recall_precision()
    #
    # print('-------------------------------\n')
    # print("eval gat dev result")
    # score_for_ss_evidence_list(gat_data, gat_data, 'gat_ss_dev_10', thresholds=0.1, top_n=[10])

    # print('-------------------------------\n')
    # bert_data = read_json_rows(config.RESULT_PATH / "bert_ss_test_10/eval_data_ss_10_test_0.1_top[10].jsonl")
    # gat_data = read_json_rows(config.RESULT_PATH / "gat_ss_test_10/eval_data_ss_10_test_0.1_top[10].jsonl")
    # print("eval bert + gat test result, calculating")
    # eval_and_save(bert_data, gat_data, 'bert_gat_merged_ss_test_5', thresholds=0.3, eval=False, save=True)
    # print("done")





