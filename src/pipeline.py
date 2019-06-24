from doc_retriv.doc_retrieve import get_doc_ids_and_fever_score
from utils.file_loader import *
from sentence_retrieval.drqa_online_tfidf import tfidf_sentense_selection

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

    # print("doc retrieving for train...")
    # doc_retriv_train_list = get_doc_ids_and_fever_score(
    #     config.FEVER_TRAIN_JSONL,
    #     config.RESULT_PATH / f"{time}/train_doc_retrive.jsonl",
    #     top_k=10,
    #     eval=True,
    #     log_file=config.LOG_PATH / f"{time}/train_doc_retrieve.log")
    # print("done with doc retrieving for train.")
    #
    # #tfidf ss
    # print("tfidf ss retrieving for dev...")
    # tfidf_ss_dev_list = tfidf_sentense_selection(doc_retriv_dev_list,
    #                                              config.RESULT_PATH / f"{time}/dev_ss_retrive_tfidf.jsonl",
    #                                              top_n=10,
    #                                              log_file=config.LOG_PATH / f"{time}/dev_ss_retrive_tfidf.log")
    # print("done with tfidf ss for dev")
    # print("tfidf ss retrieving for train...")
    # doc_retriv_train_list = str(config.RESULT_PATH / f"{time}/train_doc_retrive.jsonl")
    # tfidf_ss_train_list = tfidf_sentense_selection(doc_retriv_train_list,
    #                                                config.RESULT_PATH / f"{time}/train_ss_retrive_tfidf.jsonl",
    #                                                top_n=10,
    #                                                log_file=config.LOG_PATH / f"{time}/train_ss_retrive_tfidf.log")
    # print("done with tfidf ss for train")

if __name__ == "__main__":
    doc_retriv()

