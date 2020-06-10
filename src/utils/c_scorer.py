import six
import utils
import utils.check_sentences
from collections import Counter
import numpy as np
from utils.file_loader import save_intermidiate_results, read_json_rows
import config
from utils import fever_db
from itertools import chain
import os

SENT_LINE = '<SENT_LINE>'
SENT_DOC_TITLE = '; '

"""
This is customized scoring module
"""


def fever_doc_only(predictions, actual=None, max_evidence=5, analysis_log=None):
    '''
    This method is used to only evaluate document retrieval
    '''
    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0
    doc_id_hits = 0

    evidence_number_list = []
    error_list = []
    log_print = utils.get_adv_print_func(analysis_log.parent / f"{analysis_log.name}_f1")

    for idx, instance in enumerate(predictions):
        macro_prec = doc_macro_precision(instance, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = doc_macro_recall(instance, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

        if check_doc_id_correct(instance, actual[idx], max_evidence):
            doc_id_hits += 1
        else:
            error_list.append(instance)
        evidence_number_list.append(len(instance['predicted_docids'][:max_evidence]))

    evidence_number_counter = Counter(evidence_number_list)
    avg_length = np.mean(evidence_number_list)
    total = len(predictions)

    oracle_score = doc_id_hits / total

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)

    print("Hit:", doc_id_hits)
    print("Total:", total)
    print(evidence_number_counter.most_common())
    print("Avg. Len.:", avg_length)

    print("oracle_score, pr, rec, f1")
    print(oracle_score, pr, rec, f1)

    log_print({"Hit": doc_id_hits, "Total": total, "Avg. Len.": avg_length})
    log_print("oracle_score, pr, rec, f1")
    log_print(oracle_score, pr, rec, f1)

    save_intermidiate_results(error_list, analysis_log)

    return oracle_score, pr, rec, f1


def doc_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        # all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        # predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
        #     instance["predicted_evidence"][:max_evidence]

        # Filter out the annotation ids. We just want the evidence page and line number
        doc_evi = [e[2] for eg in instance["evidence"] for e in eg if e[3] is not None]
        pred_ids = instance["predicted_docids"] if max_evidence is None \
            else instance["predicted_docids"][:max_evidence]

        for prediction in pred_ids:
            if prediction in doc_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0


def doc_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        for evience_group in instance["evidence"]:
            # Filter out the annotation ids. We just want the evidence page and line number
            docids = [e[2] for e in evience_group]
            # Only return true if an entire group of actual sentences is in the predicted sentences
            pred_ids = instance["predicted_docids"] if max_evidence is None \
                else instance["predicted_docids"][:max_evidence]

            if all([docid in pred_ids for docid in docids]):
                return 1.0, 1.0

        return 0.0, 1.0
        # predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
        #                                                                 instance["predicted_evidence"][:max_evidence]
        #
        # for evidence_group in instance["evidence"]:
        #     evidence = [[e[2], e[3]] for e in evidence_group]
        #     if all([item in predicted_evidence for item in evidence]):
        #         We only want to score complete groups of evidence. Incomplete groups are worthless.
        # return 1.0, 1.0
        # return 0.0, 1.0
    return 0.0, 0.0


def check_predicted_evidence_format(instance):
    if 'predicted_evidence' in instance.keys() and len(instance['predicted_evidence']):
        assert all(isinstance(prediction, list)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(len(prediction) == 2
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(isinstance(prediction[0], six.string_types)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"

        assert all(isinstance(prediction[1], int)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"


def is_correct_label(instance):
    return instance["label"].upper() == instance["predicted_label"].upper()


def is_strictly_correct(instance, max_evidence=None):
    # Strict evidence matching is only for NEI class
    check_predicted_evidence_format(instance)

    if instance["label"].upper() != "NOT ENOUGH INFO" and is_correct_label(instance):
        assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

        if max_evidence is None:
            max_evidence = len(instance["predicted_evidence"])

        for evience_group in instance["evidence"]:
            # Filter out the annotation ids. We just want the evidence page and line number
            actual_sentences = [[e[2], e[3]] for e in evience_group]
            # Only return true if an entire group of actual sentences is in the predicted sentences
            if all([actual_sent in instance["predicted_evidence"][:max_evidence] for actual_sent in actual_sentences]):
                return True

    # If the class is NEI, we don't score the evidence retrieval component
    elif instance["label"].upper() == "NOT ENOUGH INFO" and is_correct_label(instance):
        return True

    return False


def is_evidence_correct(instance, max_evidence=None):
    check_predicted_evidence_format(instance)

    if instance["label"].upper() != "NOT ENOUGH INFO" and is_correct_label(instance):
        assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

        if max_evidence is None:
            max_evidence = len(instance["predicted_evidence"])

        for evience_group in instance["evidence"]:
            # Filter out the annotation ids. We just want the evidence page and line number
            actual_sentences = [[e[2], e[3]] for e in evience_group]
            # Only return true if an entire group of actual sentences is in the predicted sentences
            if all([actual_sent in instance["predicted_evidence"][:max_evidence] for actual_sent in actual_sentences]):
                return True

    # If the class is NEI, we don't score the evidence retrieval component
    elif instance["label"].upper() == "NOT ENOUGH INFO":
        return True

    return False


def evidence_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
            instance["predicted_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return this_precision, this_precision_hits if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0


def evidence_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
            instance["predicted_evidence"][:max_evidence]

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0


# Micro is not used. This code is just included to demostrate our model of macro/micro
def evidence_micro_precision(instance):
    this_precision = 0
    this_precision_hits = 0

    # We only want to score Macro F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        for prediction in instance["predicted_evidence"]:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

    return this_precision, this_precision_hits


def check_doc_id_correct(instance, actual, max_length=None):
    check_predicted_evidence_format(instance)
    predicted_docids = instance["predicted_docids"][:max_length]

    if actual["label"].upper() != "NOT ENOUGH INFO":
        for evience_group in actual["evidence"]:
            # Filter out the annotation ids. We just want the evidence page and line number
            docids = [e[2] for e in evience_group]
            # Only return true if an entire group of actual sentences is in the predicted sentences
            if max_length is None:
                max_length = 5
            # pred_ids = sorted(instance["predicted_docids"], reverse=True)[:max_length]
            # instance["predicted_docids"] = instance["predicted_docids"][:max_length]
            pred_ids = predicted_docids[:max_length]
            if all([docid in pred_ids for docid in docids]):
                return True

    elif actual["label"].upper() == "NOT ENOUGH INFO":
        return True

    return False


def check_sent_correct(instance, actual, number_of_preds):
    check_predicted_evidence_format(instance)

    # if actual is not None:

    # if actual
    if actual["label"].upper() != "NOT ENOUGH INFO":
        for evience_group in actual["evidence"]:
            # Filter out the annotation ids. We just want the evidence page and line number
            sentids = [e[2] + SENT_LINE + str(e[3]) for e in evience_group]
            # Only return true if an entire group of actual sentences is in the predicted sentences
            pred_ids = sorted(instance["predicted_sentids"], reverse=True)[:number_of_preds]
            if all([sentid in pred_ids for sentid in sentids]):
                return True

    elif actual["label"].upper() == "NOT ENOUGH INFO":
        return True

    return False

def get_ss_recall_precision(result_list):
    all_truth_s = 0
    all_true_preds = 0
    all_preds = 0
    for item in result_list:
        if item["label"].upper() != "NOT ENOUGH INFO":
            pred_ids = item["predicted_sentids"]
            all_preds += len(pred_ids)
            evi_ids = []
            for evience_group in item["evidence"]:
                # Filter out the annotation ids. We just want the evidence page and line number
                evi_ids += [e[2] + SENT_LINE + str(e[3]) for e in evience_group]
            all_truth_s += len(set(evi_ids))

            for pred_s in pred_ids:
                if pred_s in set(evi_ids):
                    all_true_preds += 1

    recall = all_true_preds / all_truth_s
    precision = all_true_preds / all_preds
    print(f"total truth/total true preds/total preds: {all_truth_s}/{all_true_preds}/{all_preds}")
    print(f"recall/precision:{recall}/{precision}")


def get_macro_ss_recall_precision(result_list):
    macro_precision = 0
    macro_precision_hits = 0
    macro_recall = 0
    macro_recall_hits = 0
    for instance in result_list:
        macro_prec = evidence_macro_precision(instance)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]
    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0
    print(f"recall/precision:{rec}/{pr}")


def get_nli_error_items(predictions, error_analysis_file):
    log_print = utils.get_adv_print_func(error_analysis_file);
    for idx, instance in enumerate(predictions):
        if check_sent_correct(instance, instance):
            if not is_correct_label(instance):
                log_print(instance)



def fever_score(predictions, actual=None, max_evidence=5, mode=None,
                error_analysis_file=None,
                verbose=False, label_it=False):
    '''
    This is a important function for different scoring.
    Pass in different parameter in mode for specific score.

    :param verbose:
    :param predictions:
    :param actual:
    :param max_evidence:
    :param mode:
    :return:
    '''

    log_print = utils.get_adv_print_func(error_analysis_file.parent / f"{error_analysis_file.name}_f1",
                                         verbose=verbose)

    correct = 0
    strict = 0
    error_count = 0

    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0
    error_items = []
    # ana_f = None
    # if error_analysis_file is not None:
    #     ana_f = open(error_analysis_file, mode='w')

    if mode is not None:
        key_list = []
        for key in mode.keys():
            key_list.append(key)

        for key in key_list:
            mode[key + '_hits'] = 0

    for idx, instance in enumerate(predictions):

        if label_it:
            instance['correct'] = False

        if mode['standard']:
            assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

            # If it's a blind test set, we need to copy in the values from the actual data
            if 'evidence' not in instance or 'label' not in instance:
                assert actual is not None, 'in blind evaluation mode, actual data must be provided'
                assert len(actual) == len(predictions), 'actual data and predicted data length must match'
                assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
                instance['evidence'] = actual[idx]['evidence']
                instance['label'] = actual[idx]['label']

            assert 'evidence' in instance.keys(), 'gold evidence must be provided'

            if is_correct_label(instance):
                correct += 1.0

                if is_strictly_correct(instance, max_evidence):
                    strict += 1.0

                    if label_it:
                        instance['correct'] = True

                # if not is_strictly_correct(instance, max_evidence):
                #     is_strictly_correct(instance, max_evidence)
                # print(instance)

            macro_prec = evidence_macro_precision(instance, max_evidence)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, max_evidence)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        if mode is not None:
            if 'check_doc_id_correct' in mode and mode['check_doc_id_correct']:
                if check_doc_id_correct(instance, actual[idx]):
                    mode['check_doc_id_correct_hits'] += 1
                else:
                    error_count += 1
                    error_items.append(instance)

            if 'check_sent_id_correct' in mode and mode['check_sent_id_correct']:
                if check_sent_correct(instance, actual[idx], max_evidence):
                    mode['check_sent_id_correct_hits'] += 1
                else:
                    error_count += 1
                    if 'predicted_evidence' in instance.keys():
                        error_items.append(get_pred_instantce_details(instance))
                    else:
                        error_items.append(instance)

    log_print("Error count:", error_count)
    total = len(predictions)

    print("Total:", total)
    print("Correct:", correct)
    print(f"Strict total/percentage: {strict}/{strict/total}")
    print("Error count:", error_count)
    log_print("Total:", total)
    log_print("Correct:", correct)
    log_print(f"Strict total/percentage: {strict}/{strict/total}")

    for k, v in mode.items():
        if k.endswith('_hits'):
            log_print(k, v, v / total)
            print(k, v, v / total)

    strict_score = strict / total
    acc_score = correct / total

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    if pr + rec == 0:
        f1 = 0
    else:
        f1 = 2.0 * pr * rec / (pr + rec)

    log_print("pr: ",  pr)
    log_print("recall: ", rec)
    log_print("f1:", f1)
    print("pr: ",  pr)
    print("recall: ", rec)
    print("f1:", f1)
    save_intermidiate_results(error_items, error_analysis_file)

    return strict_score, acc_score, pr, rec, f1


def get_evid_text(evids_list):
    cursor, conn = fever_db.get_cursor()
    current_evidence_text = []
    for e in evids_list:
        doc_id, line_num = e
        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)
        current_evidence_text.append(e_text)
    cursor.close()
    conn.close()
    return current_evidence_text


def get_pred_instantce_details(item):
    el = utils.check_sentences.check_and_clean_evidence(item)
    el_l = list(chain.from_iterable(el))
    evidence_text = get_evid_text(el_l)
    item['evidence_texts'] = evidence_text
    predl = utils.check_sentences.get_predicted_evidence(item)
    pred_text = get_evid_text(list(chain.from_iterable(predl)))
    item['predicted_texts'] = pred_text
    return item


def delete_label(d_list):
    for item in d_list:
        if 'label' in item:
            del item['label']
        if 'evidence' in item:
            del item['evidence']


def nei_stats(predictions, actual=None):
    '''
    This is a important function for different scoring.
    Pass in different parameter in mode for specific score.

    :param verbose:
    :param predictions:
    :param actual:
    :param max_evidence:
    :param mode:
    :return:
    '''
    total = 0
    empty = 0

    empty_support = 0
    empty_refutes = 0
    total_s = 0
    total_r = 0

    for idx, instance in enumerate(predictions):

        if actual[idx]['label'] == "NOT ENOUGH INFO":
            if len(instance['predicted_evidence']) == 0:
                empty += 1
            total += 1
        elif actual[idx]['label'] == 'SUPPORTS':
            if len(instance['predicted_evidence']) == 0:
                empty_support += 1
            total_s += 1
        elif actual[idx]['label'] == 'REFUTES':
            if len(instance['predicted_evidence']) == 0:
                empty_refutes += 1
            total_r += 1

    print("NEI", empty, total, empty / total)
    print("SUP", empty_support, total_s, empty_support / total_s)
    print("REF", empty_refutes, total_r, empty_refutes / total_r)

    return empty, total

# --------------------------------------------------
# Best Acc: 0.7035703570357036
# Best Acc Ind: [0, 1, 2, 3, 4]
# --------------------------------------------------
# --------------------------------------------------
# Best sAcc: 0.6693169316931693
# Best sAcc Ind: [0, 1, 2, 3, 4]
# --------------------------------------------------
# --------------------------------------------------
# Best Acc: 0.7047204720472047
# Best Acc Ind: [0, 1, 2, 4, 11]
# --------------------------------------------------

if __name__ == "__main__":
    # eval_mode = {'check_sent_id_correct': True, 'standard': False}
    # fever_score(read_json_rows(config.RESULT_PATH / 'dev_pred_ss_2019_07_31/eval_data_ss_dev_0.5_top5.jsonl'),
    #             read_json_rows(config.DOC_RETRV_DEV),
    #             mode=eval_mode,
    #             error_analysis_file=config.LOG_PATH / 'dev_pred_f1_error_items.log')
    # upstream_data = read_json_rows(config.RESULT_PATH / 'eval_data_nli_dev_0.5_top5.jsonl')
    # original_data = read_json_rows(config.FEVER_DEV_JSONL)
    # nei_stats(upstream_data, original_data)
    upstream_data5 = read_json_rows(config.RESULT_PATH / 'dev_pred_ss_2020_03_15_16:50:23/eval_data_ss_5_dev_0.4_top[10, 5].jsonl')
    upstream_data10 = read_json_rows(config.RESULT_PATH / 'dev_pred_ss_2020_03_15_16:50:23/eval_data_ss_10_dev_0.4_top[10, 5].jsonl')
    # get_ss_recall_precision(upstream_data5)
    get_macro_ss_recall_precision(upstream_data5)
    # get_ss_recall_precision(upstream_data10)
    get_macro_ss_recall_precision(upstream_data10)
