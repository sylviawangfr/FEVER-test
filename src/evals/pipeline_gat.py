from utils.file_loader import read_json_rows, save_intermidiate_results
from sample_for_nli_esim.tf_idf_sample_v1_0 import convert_evidence2scoring_format
import utils.common_types as model_para
from BERT_test.ss_eval import ss_f1_score_and_save
import config
from utils import c_scorer
from ss_combined.ss_bert_gat import eval_and_save, score_for_ss_evidence_list
from BERT_test.bert_data_processor import *
from BERT_test.nli_eval import eval_nli_and_save
from evals.submmit import create_submmission

def dev_gat_eval():
    threshold = 0.3
    bert_data = read_json_rows(config.RESULT_PATH / "bert_ss_dev_10/eval_data_ss_10_dev_0.1_top[10].jsonl")
    gat_data = read_json_rows(config.RESULT_PATH / "gat_ss_dev_10/eval_data_ss_10_dev_0.1_top[10].jsonl")
    print("eval bert + gat dev result")
    eval_and_save(bert_data, gat_data, 'bert_gat_merged_ss_dev_5', thresholds=threshold, save=True)

    print('-------------------------------\n')
    print("eval bert dev result")
    score_for_ss_evidence_list(bert_data, bert_data, 'bert_ss_dev_10', thresholds=threshold, top_n=[10, 5])

    print('-------------------------------\n')
    print("eval gat dev result")
    score_for_ss_evidence_list(gat_data, gat_data, 'gat_ss_dev_10', thresholds=threshold, top_n=[10, 5])

    print('-------------------------------\n')
    paras = bert_para.PipelineParas()
    paras.pred = True
    paras.mode = 'dev'
    paras.original_data = read_json_rows(config.FEVER_DEV_JSONL)
    paras.upstream_data = read_json_rows(config.RESULT_PATH / "bert_gat_merged_ss_dev_5/eval_data_bert_gat_ss_5_dev_0.1_top[5].jsonl")
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    paras.output_folder = 'nli_dev_bert_gat'
    eval_nli_and_save(paras)


def test_eval():
    threshold = 0.3
    print('-------------------------------\n')
    bert_data = read_json_rows(config.RESULT_PATH / "bert_ss_test_10/eval_data_ss_10_test_0.1_top[10].jsonl")
    gat_data = read_json_rows(config.RESULT_PATH / "gat_ss_test_10/eval_data_ss_10_test_0.1_top[10].jsonl")
    print("eval bert + gat test result, calculating")
    eval_and_save(bert_data, gat_data, 'bert_gat_merged_ss_test_5', thresholds=threshold, eval=False, save=True)

    print('-------------------------------\n')
    paras = bert_para.PipelineParas()
    paras.pred = True
    paras.mode = 'test'
    paras.original_data = read_json_rows(config.FEVER_TEST_JSONL)
    paras.upstream_data = read_json_rows(config.RESULT_PATH / "bert_gat_merged_ss_test_5/eval_data_bert_gat_ss_5_test_0.1_top[5].jsonl")
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    paras.output_folder = 'nli_test_bert_gat'
    eval_nli_and_save(paras)
    input_data = read_json_rows(config.RESULT_PATH / 'nli_test_bert_gat/eval_data_nli_test_0.5_top[5].jsonl')
    create_submmission(input_data)


if __name__ == '__main__':
    dev_gat_eval()
