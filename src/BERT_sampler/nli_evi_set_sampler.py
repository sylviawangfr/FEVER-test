from utils.check_sentences import Evidences, sids_to_tuples
from tqdm import tqdm
from utils.resource_manager import FeverDBResource
from data_util.data_preperation.tokenize_fever import easy_tokenize
from utils import c_scorer, text_clean
from utils import fever_db
from utils.file_loader import read_json_rows


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
    with tqdm(total=len(d_list), desc=f"sampling nli data") as pbar:
        for idx, item in enumerate(d_list):
            sampled_e_list = nli_evi_set_sampler(item)
            sampled_sids = item['nli_sids']
            for idx, sampled_evidence in enumerate(sampled_e_list):
                evidence_text = evidence_list_to_text(sampled_evidence,
                                                      contain_head=True, id_tokenized=False)
                new_item = dict()
                new_item['claim'] = item['claim']
                new_item['id'] = str(item['id']) + '#' + str(idx)
                new_item['evid'] = evidence_text
                new_item['sids'] = sampled_sids[idx]
                if mode == 'pred':
                    # not used, but to avoid example error
                    new_item['label'] = 'NOT ENOUGH INFO'
                else:
                    new_item['label'] = item['label']
                sampled_data_list.append(new_item)
                pbar.update(1)
    print(f"Sampled evidences: {len(sampled_data_list)}")
    return sampled_data_list


def evidence_list_to_text(evidences, contain_head=True):
    current_evidence_text = []
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))
    cur_head = 'DO NOT INCLUDE THIS FLAG'
    db = FeverDBResource()
    cursor, conn = db.get_cursor()
    for doc_id, line_num in evidences:
        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)
        if contain_head and cur_head != doc_id:
            cur_head = doc_id
            doc_id_natural_format = text_clean.convert_brc(doc_id).replace('_', ' ')
            t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            if line_num != 0:
                current_evidence_text.append(f"{t_doc_id_natural_format}{c_scorer.SENT_DOC_TITLE}")
        current_evidence_text.append(e_text)
    return ' '.join(current_evidence_text)


if __name__ == '__main__':
    pass