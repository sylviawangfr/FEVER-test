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

def test():
    l = ['1', '2', '3']
    t = '\n'.join(l)
    t = t + '\n'

    print("start")
    print(t)
    print('end')
    save_file(t, config.RESULT_PATH / "t.txt")


if __name__ == "__main__":
    convert_wiki_to_bert_format()

