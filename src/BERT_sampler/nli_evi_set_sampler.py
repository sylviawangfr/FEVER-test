from utils.check_sentences import Evidences, sids_to_tuples
from tqdm import tqdm
from data_util.data_preperation.tokenize_fever import easy_tokenize
from utils import c_scorer, text_clean
from utils.file_loader import read_json_rows
from utils import fever_db, check_sentences
from typing import List
import BERT_sampler.nli_nn_sampler as nli_nn_sampler
import config


# mode in ['train', 'pred', 'eval']
def nli_evi_set_sampler(item_with_nli_evi_sets):
    candidate_evi_sid_set = item_with_nli_evi_sets['nli_sids']
    tuple_set = [sids_to_tuples(i) for i in candidate_evi_sid_set]
    candidate_evi_set = [Evidences(i) for i in tuple_set]
    return candidate_evi_set


# mode in ['train', 'pred', 'eval']
def get_sample_data(upstream_data, data_from_pred=True, mode='pred'):
    if isinstance(upstream_data, list):
        d_list = upstream_data
    else:
        d_list = read_json_rows(upstream_data)
    sampled_data_list = []
    if data_from_pred:
        with tqdm(total=len(d_list), desc=f"sampling nli data") as pbar:
            for idx, item in enumerate(d_list):
                if item['verifiable'] == "VERIFIABLE":
                    assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
                    e_set = check_sentences.check_and_clean_evidence(item)
                if data_from_pred:
                    sampled_e_list = nli_evi_set_sampler(item)
                    sampled_sids = item['nli_sids']
                    for idx, sampled_evidence in enumerate(sampled_e_list):
                        evidence_text = nli_nn_sampler.evidence_list_to_text(sampled_evidence, contain_head=True)
                        new_item = dict()
                        new_item['claim'] = item['claim']
                        new_item['id'] = str(item['id']) + '#' + str(idx)
                        new_item['evid'] = evidence_text
                        new_item['sids'] = sampled_sids[idx]
                        if mode == 'pred':
                            # not used, but to avoid example error
                            new_item['label'] = 'NOT ENOUGH INFO'
                        else:
                            if item['label'] != 'NOT ENOUGH INFO' and is_evidence_correct(e_set, sampled_evidence):
                                new_item['label'] = item['label']
                            else:
                                new_item['label'] = 'NOT ENOUGH INFO'
                        sampled_data_list.append(new_item)
        pbar.update(1)
    else:
        sampled_data_list = nli_nn_sampler.get_sample_data(upstream_data, data_from_pred=False, mode='train')
    print(f"Sampled evidences: {len(sampled_data_list)}")
    return sampled_data_list


def is_evidence_correct(all_evis: List[Evidences], to_check_evi: Evidences):
    # Only return true if an entire group of actual sentences is in the predicted sentences
    if any([actual_evi in to_check_evi for actual_evi in all_evis]):
        return True
    return False


if __name__ == '__main__':
    d = read_json_rows(config.RESULT_PATH / "hardset2021/bert_ss_0.4_10.jsonl")
    samples = get_sample_data(d, data_from_pred=False, mode='eval')
    nli_nn_sampler.eval_samples(samples)