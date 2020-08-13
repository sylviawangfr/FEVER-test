from tfidf_model.drqa_online_tfidf import tfidf_sentense_selection
import config
from utils.file_loader import get_current_time_str


def get_ss_tfidf(doc_retriv_l, out_file_path):
    print("tfidf ss retrieving...")
    tfidf_ss_dev_list = tfidf_sentense_selection(doc_retriv_l,
                                                 out_file_path,
                                                 top_n=10,
                                                 log_file=config.LOG_PATH / f"ss_retrive_tfidf_{get_current_time_str()}.log")
    return tfidf_ss_dev_list

