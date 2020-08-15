from utils.check_sentences import check_doc_id
from utils.fever_db import *
from utils.file_loader import save_file
from utils.text_clean import convert_brc


def convert_wiki_to_bert_format():
    wiki_data = ''
    cursor, conn = get_cursor()
    doc_ids = get_all_doc_ids(cursor)
    for i in tqdm(doc_ids):
        sentences, id_list= get_all_sent_by_doc_id(cursor, i)
        sentences = [convert_brc(i) for i in sentences]
        one_doc = '\n'.join(sentences)
        wiki_data = wiki_data + one_doc + "\n\n"
    save_file(wiki_data, config.RESULT_PATH / "wiki_to_bert.txt")
    cursor.close()
    conn.close()


def convert_wiki_to_bert_format_retri_doc():
    doc_train = read_json_rows(config.DOC_RETRV_TRAIN)[0:10]
    doc_dev = read_json_rows(config.DOC_RETRV_DEV)[0:10]
    doc_test = read_json_rows(config.DOC_RETRV_TEST)[0:10]
    all_docs = get_doc_ids(doc_train) | get_doc_ids(doc_dev) | get_doc_ids(doc_test)
    all_docs.remove(None)
    wiki_data = ''
    cursor, conn = get_cursor()
    for i in tqdm(all_docs):
        sentences, id_list= get_all_sent_by_doc_id(cursor, i)
        sentences = [convert_brc(i) for i in sentences]
        one_doc = '\n'.join(sentences)
        wiki_data = wiki_data + one_doc + "\n\n"
    save_file(wiki_data, config.RESULT_PATH / "wiki_to_bert.txt")
    cursor.close()
    conn.close()

def get_doc_ids(doc_data):
    all_doc_ids = set()
    for item in tqdm(doc_data):
        doc_ids = item["predicted_docids"]
        all_doc_ids = all_doc_ids | set(doc_ids)
        if 'evidence' in item.keys() and item['evidence'] is not None:
            # ground truth
            e_set = check_doc_id(item)
            all_doc_ids = all_doc_ids | e_set

    return all_doc_ids

def test():
    l = ['1', '2', '3']
    t = '\n'.join(l)
    t = t + '\n'

    print("start")
    print(t)
    print('end')
    save_file(t, config.RESULT_PATH / "t.txt")


if __name__ == "__main__":
    convert_wiki_to_bert_format_retri_doc()

