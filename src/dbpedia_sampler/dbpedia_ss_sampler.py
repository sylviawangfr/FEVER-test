import itertools

import BERT_sampler.ss_sampler as ss_sampler
import log_util
import utils.check_sentences
import utils.common_types as bert_para
from dbpedia_sampler import dbpedia_subgraph
from bert_serving.client import BertClient
from utils import fever_db, c_scorer
from utils.file_loader import *
from utils.iter_basket import BasketIterable
from torch.utils.data import DataLoader
from utils.text_clean import convert_brc
from memory_profiler import profile
import gc
from collections import Counter


log = log_util.get_logger("dbpedia_ss_sampler")


# @profile
def get_tfidf_sample(paras: bert_para.PipelineParas):
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
    for idx, item in enumerate(d_list):
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

        claim_dict = dbpedia_subgraph.construct_subgraph_for_claim(convert_brc(item['claim']), bc=paras.bert_client)
        example_l = []
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(item['id']) + "<##>" + str(sent_item['sid'])
            if 'label' in item.keys():
                sent_item['claim_label'] = item['label']
            sentence = convert_brc(sent_item['text'])
            doc_title = sent_item['sid'].split(c_scorer.SENT_LINE)[0].replace("_", " ")
            doc_title = convert_brc(doc_title)
            if sentence.startswith(f"{doc_title} - "):
                sentence = sentence.replace(f"{doc_title} - ", "")
            sent_item['graph'] = dbpedia_subgraph.construct_subgraph_for_candidate(claim_dict, sentence, doc_title, bc=paras.bert_client)
            example_l.append(sent_item)

        one_full_example['claim_links'] = claim_dict['graph']
        one_full_example['examples'] = example_l
        dbpedia_examples_l.append(one_full_example)

    cursor.close()
    conn.close()
    # ss_sampler.count_truth_examples([j for i in dbpedia_examples_l for j in i['examples']])
    # logger.info(np.sum(count_truth))
    return dbpedia_examples_l


def get_full_list_from_upstream_ss(paras: bert_para.PipelineParas):
    if isinstance(paras.upstream_data, list):
        d_list = paras.upstream_data
    else:
        d_list = read_json_rows(paras.upstream_data)

    dbpedia_examples_l = []
    cursor, conn = fever_db.get_cursor()
    err_log_f = config.LOG_PATH / f"{utils.get_current_time_str()}_dbpedia_sample.log"
    for idx, item in enumerate(d_list):
        one_full_example = dict()
        one_full_example['claim'] = item['claim']
        one_full_example['id'] = item['id']
        predicted_evidence = item["predicted_sentids"]
        num_envs = 0
        r_list = []
        id_list = []
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

        all_sent_list = ss_sampler.convert_to_formatted_sent(zipped_s_id_list, None, contain_head=True,
                                                             id_tokenized=True)
        claim_dict = dbpedia_subgraph.construct_subgraph_for_claim(convert_brc(item['claim']), bc=paras.bert_client)
        example_l = []
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(item['id']) + "<##>" + str(sent_item['sid'])
            if 'label' in item.keys():
                sent_item['claim_label'] = item['label']
            sentence = convert_brc(sent_item['text'])
            doc_title = sent_item['sid'].split(c_scorer.SENT_LINE)[0].replace("_", " ")
            doc_title = convert_brc(doc_title)
            if sentence.startswith(f"{doc_title} - "):
                sentence = sentence.replace(f"{doc_title} - ", "")
            sent_item['graph'] = dbpedia_subgraph.construct_subgraph_for_candidate(claim_dict, sentence, doc_title,
                                                                                   bc=paras.bert_client)
            example_l.append(sent_item)
        one_full_example['claim_links'] = claim_dict['graph']
        one_full_example['examples'] = example_l
        dbpedia_examples_l.append(one_full_example)
    cursor.close()
    conn.close()
    return dbpedia_examples_l


def collate(samples):
    return samples


# @profile
def convert_to_graph_sampler(upstream_data, output_file, pred=False):
    batch_size = 10
    dt = get_current_time_str()
    paras = bert_para.PipelineParas()
    paras.pred = pred
    paras.bert_client = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    sample_dataloader = DataLoader(upstream_data, batch_size=batch_size, collate_fn=collate)
    # sample_dataloader = BasketIterable(tfidf_data, batch_size)
    batch = 0
    with tqdm(total=len(sample_dataloader), desc=f"Sampling") as pbar:
        for batched_sample in sample_dataloader:
            paras.upstream_data = batched_sample
            if pred:
                paras.sample_n = 10
                samples = get_full_list_from_upstream_ss(paras)
            else:
                paras.sample_n = 3
                samples = get_tfidf_sample(paras)
            num = batch * batch_size + len(batched_sample)
            log.info(f"total count: {num}")
            save_and_append_results(samples, num, output_file,
                                    config.LOG_PATH / f"sample_ss_graph_{dt}.log")
            pbar.update(1)
            batch += 1
    paras.bert_client.close()
    return


if __name__ == '__main__':
    # multi_thread_sampler()
    dev_data = read_json_rows(config.RESULT_PATH / "bert_ss_dev_10/eval_data_ss_10_dev_0.1_top[10].jsonl")[5000:10000]
    # tfidf_dev_data = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")[6980:13000]
    # dev_data = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")[0:1]
    # tfidf_dev_data = tfidf_dev_data[13000:len(tfidf_dev_data)]
    convert_to_graph_sampler(dev_data, config.RESULT_PATH / "sample_ss_graph_dev_pred" / f"2_{get_current_time_str()}.jsonl", pred=True)
    # # #
    #
    # tfidf_train_data = read_json_rows(config.RESULT_PATH / "train_s_tfidf_retrieve.jsonl")[93420:100000]
    # tfidf_train_data = read_json_rows(config.RESULT_PATH / "train_s_tfidf_retrieve.jsonl")[84840:90000]
    # tfidf_train_data = read_json_rows(config.RESULT_PATH / "train_s_tfidf_retrieve.jsonl")[114320:120000]
    # tfidf_train_data = read_json_rows(config.RESULT_PATH / "train_s_tfidf_retrieve.jsonl")
    # tfidf_train_data = tfidf_train_data[140000:len(tfidf_train_data)]
    # tfidf_train_data = read_json_rows(config.RESULT_PATH / "train_s_tfidf_retrieve.jsonl")[120000:130000]
    # tfidf_train_data = read_json_rows(config.RESULT_PATH / "train_s_tfidf_retrieve.jsonl")[130000:140000]
    # tfidf_to_graph_sampler(tfidf_train_data)
    # # # print(globals())
    # print(json.dumps(globals(), indent=1))
    # test_memory()
