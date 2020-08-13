from doc_retriv.doc_retrieve import get_doc_ids_and_fever_score
import config
from utils.file_loader import get_current_time_str


def get_doc(input_data_path, output_path):
    print("doc retrieving for dev...")
    dt = get_current_time_str()
    doc_retriv_list = get_doc_ids_and_fever_score(
        input_data_path,
        output_path,
        top_k=10,
        eval=True,
        log_file=config.LOG_PATH / f"doc_retriv_{dt}.log")

    print("done with doc retrieving")
    return doc_retriv_list

