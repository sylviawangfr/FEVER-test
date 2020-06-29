import utils.check_sentences
import itertools
from utils.file_loader import *
import utils.common_types as bert_para
import BERT_sampler.ss_sampler as ss_sampler
from utils import fever_db, c_scorer
from dbpedia_sampler import dbpedia_subgraph
import log_util

log = log_util.get_logger("dbpedia_ss_sampler")

def get_tfidf_sample(paras: bert_para.BERT_para):
    """
    This method will select all the sentence from upstream tfidf ss retrieval and label the correct evident as true for nn model
    :param tfidf_ss_data_file: Remember this is result of tfidf ss data with original format containing 'evidence' and 'predicted_evidence'

    :return:
    """

    if not isinstance(paras.upstream_data, list):
        d_list = read_json_rows(paras.upstream_data)
    else:
        d_list = paras.upstream_data

    cursor, conn = fever_db.get_cursor()
    err_log_f = config.LOG_PATH / f"{get_current_time_str()}_analyze_sample.log"
    count_truth = []
    dbpedia_examples_l = []
    for item in tqdm(d_list):
        one_full_example = dict()
        one_full_example['claim'] = item['claim']
        one_full_example['id'] = item['id']
        predicted_evidence = item["predicted_sentids"]
        ground_truth = item['evidence']
        if paras.pred:            # If pred, then reset to not containing ground truth evidence.
            all_evidence_set = None
            r_list = []
            id_list = []
        else:
            if ground_truth is not None and len(ground_truth) > 0:
                e_list = utils.check_sentences.check_and_clean_evidence(item)
                all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))
            else:
                all_evidence_set = None
            r_list = []
            id_list = []

            if all_evidence_set is not None:
                for doc_id, ln in all_evidence_set:
                    _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
                    r_list.append(text)
                    id_list.append(doc_id + '(-.-)' + str(ln))

        num_envs = 0 if all_evidence_set is None else len(all_evidence_set)
        count_truth.append(num_envs)
        for pred_item in predicted_evidence:
            if num_envs >= paras.sample_n:
                break
            doc_id, ln = pred_item.split(c_scorer.SENT_LINE)[0], int(pred_item.split(c_scorer.SENT_LINE)[1])
            tmp_id = doc_id + '(-.-)' + str(ln)
            if not tmp_id in id_list:
                _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
                r_list.append(text)
                id_list.append(tmp_id)
                num_envs = num_envs + 1

        if not (len(id_list) == len(set(id_list)) or len(r_list) == len(id_list)):
            utils.get_adv_print_func(err_log_f)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = ss_sampler.convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
                                                  id_tokenized=True)

        claim_dict = dbpedia_subgraph.construct_subgraph_for_claim(text_clean.convert_brc(item['claim']))
        example_l = []
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(item['id']) + "<##>" + str(sent_item['sid'])
            if 'label' in item.keys():
                sent_item['claim_label'] = item['label']
            sentence = text_clean.convert_brc(sent_item['text'])
            doc_title = sent_item['sid'].split(c_scorer.SENT_LINE)[0].replace("_", " ")
            doc_title = text_clean.convert_brc(doc_title)
            sent_item['graph'] = dbpedia_subgraph.construct_subgraph_for_candidate(claim_dict, sentence, doc_title)
            example_l.append(sent_item)

        one_full_example['claim_links'] = claim_dict['graph']
        one_full_example['examples'] = example_l
        dbpedia_examples_l.append(one_full_example)

    cursor.close()
    conn.close()
    # ss_sampler.count_truth_examples([j for i in dbpedia_examples_l for j in i['examples']])
    # logger.info(np.sum(count_truth))
    return dbpedia_examples_l


def prepare_train_data_filter_full_list():
    paras2 = bert_para.BERT_para()
    train_upstream_data = read_json_rows(config.DOC_RETRV_TRAIN)[5:15]
    paras2.upstream_data = train_upstream_data
    paras2.pred = False
    paras2.post_filter_prob = 0.05
    complete_upstream_train_data = ss_sampler.get_full_list_sample(paras2)
    return complete_upstream_train_data


def prepare_train_data_filter_tfidf(tfidf_data):
    paras = bert_para.BERT_para()
    # all_data = read_json_rows(config.RESULT_PATH / "train_s_tfidf_retrieve.jsonl")[0:100]
    data_len = len(tfidf_data)
    paras.sample_n = 3
    paras.pred = False
    bulk_size = 3
    start = 0
    while start < data_len:
        end = start + bulk_size if start + bulk_size < data_len else data_len
        paras.upstream_data = tfidf_data[start:end]
        sample_tfidf = get_tfidf_sample(paras)
        dt = get_current_time_str()
        save_and_append_results(sample_tfidf, config.RESULT_PATH / f"sample_ss_graph_{dt}.jsonl",
                                config.LOG_PATH / f"sample_ss_graph_{dt}.log")
        log.info(f"Finished total count: {end}")
        if end == data_len:
            break
        else:
            start = end
    return sample_tfidf


def construct_graphs_for_example(example):
    pass


def cache_temp_graph_result_to_file():
    pass


if __name__ == '__main__':
    tfidf_dev_data = read_json_rows(config.RESULT_PATH / "ss_tfidf_error_data.jsonl")
    prepare_train_data_filter_tfidf(tfidf_dev_data)

