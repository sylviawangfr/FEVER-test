from BERT_sampler.ss_sampler import *
from utils.file_loader import *


def prepare_ss_sample_for_ernie(json_data, filename):
    text = f"text_a\ttext_b\tlabel\n"
    for item in json_data:
        s1 = text_clean.convert_brc(item['query'])
        s2 = text_clean.convert_brc(item['text'])
        label = 1 if item['selection_label'] == 'true' else 0
        text = text + f"{s1}\t{s2}\t{label}\n"
    save_file(text, filename)
    return


if __name__ == '__main__':
    train_data = read_json_rows(config.RESULT_PATH / "tfidf" / "train_2019_06_15_15:48:58.jsonl")
    dev_data = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")
    sample_list1 = get_tfidf_sample(train_data)
    prepare_ss_sample_for_ernie(sample_list1, config.RESULT_PATH / "ernie_ss_train.tsv")
    sample_list2 = get_tfidf_sample(dev_data)
    prepare_ss_sample_for_ernie(sample_list2, config.RESULT_PATH / "ernie_ss_dev.tsv")
