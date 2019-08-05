from doc_retriv.doc_retrieve import get_doc_ids_and_fever_score
from utils.file_loader import *
from sentence_retrieval_esim.drqa_online_tfidf import tfidf_sentense_selection
from BERT_test.ss_eval import *
from BERT_test.nli_eval import *

time = "june22"


def doc_retriv():
    # doc retrieve
    # time = get_current_time_str()

    print("doc retrieving for dev...")
    doc_retriv_dev_list = get_doc_ids_and_fever_score(
        config.FEVER_DEV_JSONL,
        config.RESULT_PATH / f"{time}/dev_doc_retrive.jsonl",
        top_k=10,
        eval=True,
        log_file=config.LOG_PATH / f"{time}/dev_doc_retrieve.log")
    print("done doc retrieving for dev.")

    print("doc retrieving for train...")
    doc_retriv_train_list = get_doc_ids_and_fever_score(
        config.FEVER_TRAIN_JSONL,
        config.RESULT_PATH / f"{time}/train_doc_retrive.jsonl",
        top_k=10,
        eval=True,
        log_file=config.LOG_PATH / f"{time}/train_doc_retrieve.log")
    print("done with doc retrieving for train.")

    #tfidf ss
    print("tfidf ss retrieving for dev...")
    tfidf_ss_dev_list = tfidf_sentense_selection(doc_retriv_dev_list,
                                                 config.RESULT_PATH / f"{time}/dev_ss_retrive_tfidf.jsonl",
                                                 top_n=10,
                                                 log_file=config.LOG_PATH / f"{time}/dev_ss_retrive_tfidf.log")
    print("done with tfidf ss for dev")
    print("tfidf ss retrieving for train...")
    doc_retriv_train_list = str(config.RESULT_PATH / f"{time}/train_doc_retrive.jsonl")
    tfidf_ss_train_list = tfidf_sentense_selection(doc_retriv_train_list,
                                                   config.RESULT_PATH / f"{time}/train_ss_retrive_tfidf.jsonl",
                                                   top_n=10,
                                                   log_file=config.LOG_PATH / f"{time}/train_ss_retrive_tfidf.log")
    print("done with tfidf ss for train")


def get_doc(input_data_path):
    print("doc retrieving for dev...")
    doc_retriv_dev_list = get_doc_ids_and_fever_score(
        input_data_path,
        config.RESULT_PATH / f"doc_{input_data_path.name}",
        top_k=10,
        eval=True,
        log_file=config.LOG_PATH / f"doc_{input_data_path.name}.log")

    print("done doc retrieving for dev.")


def pred_ss(input_data_path, origin_data_path, output_file):
    paras = bert_para.BERT_para()
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"

    paras.output_folder = output_file
    paras.sample_n = 5
    paras.mode = 'dev'
    paras.pred = True
    paras.original_data = read_json_rows(origin_data_path)
    paras.upstream_data = read_json_rows(input_data_path)
    pred_ss_and_save(paras)


def nli(input_data_path, origin_data_path, output_file):
    paras = bert_para.BERT_para()
    paras.pred = True
    paras.mode = 'dev'
    paras.original_data = read_json_rows(origin_data_path)
    paras.upstream_data = read_json_rows(input_data_path)
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    paras.output_folder = output_file
    eval_nli_and_save(paras)


if __name__ == "__main__":
    input_file = config.RESULT_PATH / "doc_hardset.jsonl"
    ss_file = "pred_ss_" + input_file.name
    nli_file = "nli_" + input_file.name
    pred_ss(input_file, input_file, ss_file)
    nli(ss_file, input_file, nli_file)



