import unittest
from utils.fever_db import *
from utils.sqlite_queue import *
from ES.es_search import *


class TestDB(unittest.TestCase):

    def test_multithread_db(self):
        d = SQLiteUtil(str(config.FEVER_DB))
        d_hash = d.__hash__()
        r_list = []
        thread_exe(lambda i: r_list.append(SQLiteUtil(str(config.FEVER_DB)).__hash__()), range(0,10), 10, "test db threads")
        assert all(item == d_hash for item in r_list)

    def test_compare_threads(self):
        one_conn = SQLiteUtil(str(config.FEVER_DB))
        list_d = read_json_rows(config.DATA_ROOT / "wiki-pages/wiki-001.jsonl")[1:10000]
        list_id = [item['id'] for item in list_d]
        qr = []
        time1_s = datetime.datetime.now()
        thread_exe(lambda x: qr.append(x) if get_all_sent_by_doc_id_mutithread(one_conn, x) == ([], []) else None,
                   list_id, 2000, "testing multithread query")
        print(datetime.datetime.now() - time1_s)

        cursor, conn = get_cursor()
        qr_single = []
        time2_s = datetime.datetime.now()
        for i in tqdm(list_id):
            r = get_all_sent_by_doc_id(cursor, i)
            if r == ([], []):
                qr_single.append(i)
        conn.close()
        print(datetime.datetime.now() - time2_s)
        assert len(qr) == len(qr_single)


    # def test_es_query(self):
    #     list_d = read_json_rows(config.DATA_ROOT / "wiki-pages/wiki-001.jsonl")[1:10000]
    #     list_id = [item['id'] for item in list_d]
    #     one_conn = SQLiteUtil(str(config.FEVER_DB))
    #     qr = []
    #     thread_exe(lambda x: qr.append(x) if get_all_sent_by_doc_id_mutithread(one_conn, x) == ([], []) else None,
    #                list_id, 2000, "testing multithread query")
    #     print("done with sqlite3 query")
    #
    #     qr_es = []
    #     thread_exe(lambda x: qr_es.append(x) if get_all_sent_by_doc_id_es(x) == ([], []) else None,
    #                list_id, 2000, "testing multithread query by es")
    #     print("done with es query")
    #
    #     assert len(qr) == len(qr_es)


if __name__ == '__main__':
    unittest.main()
