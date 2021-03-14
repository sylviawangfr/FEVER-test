import copy
from BERT_test.ss_eval import *
from utils.resource_manager import *
from collections import Counter
from doc_retriv.doc_retrieve_extend import *
from doc_retriv.ss import *


def eval_tri_ss(data_origin, data_tri):
    for idx, example in enumerate(data_tri):
        tri_s_l = example['triple_sentences']
        sids = [sid for tri in tri_s_l.values() for sid in tri]
        sids = list(set(sids))
        docids = [i.split(c_scorer.SENT_LINE2)[0] for i in sids]
        docids = list(set(docids))
        formated_sid = [i.replace(c_scorer.SENT_LINE2, c_scorer.SENT_LINE) for i in sids]
        data_origin[idx]['predicted_sentids'] = formated_sid
        data_origin[idx]['predicted_evidence'] = convert_evidence2scoring_format(formated_sid)
        data_origin[idx]['predicted_docs'] = docids
    hit_eval(data_origin, 10, with_doc=True)
    c_scorer.get_macro_ss_recall_precision(data_origin, 10)


def hit_eval(data, max_evidence, with_doc=False):
    one_hit = 0
    total = 0
    for idx, instance in enumerate(data):
        if instance["label"].upper() != "NOT ENOUGH INFO":
            total += 1.0
            all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

            predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                instance["predicted_evidence"][:max_evidence]

            for prediction in predicted_evidence:
                if prediction in all_evi:
                    one_hit += 1.0
                    break
    print(f"at least one ss hit: {one_hit}, total: {total}, rate: {one_hit / total}")  # 1499, 1741, 0.86
    if with_doc:
        one_hit = 0
        for instance in data:
            all_evi_docs = [e[2] for eg in instance["evidence"] for e in eg if e[3] is not None]
            all_evi_docs = list(set(all_evi_docs))
            predicted_docid = instance['predicted_docs']
            for prediction in predicted_docid:
                if prediction in all_evi_docs:
                    one_hit += 1.0
                    break
    print(f"at least one doc hit: {one_hit}, total: {total}, rate: {one_hit / total}")  #

def eval_dbpedia_links(graph_data):
    entity_one_hit = 0
    for i in graph_data:
        linked_phrases = i['claim_dict']['linked_phrases_l']
        linked_entities = [ent['text'] for ent in linked_phrases]
        all_evi_docs = list(set([convert_brc(e[2].lower()).replace("_", " ") for eg in i["evidence"] for e in eg if e[3] is not None]))
        for ll in linked_entities:
            if len(list(filter(lambda x: ll.lower() in x, all_evi_docs))) > 0:
                entity_one_hit += 1
                break
    print(f"at least one entity doc hit: {entity_one_hit}, total: {len(graph_data)}, rate: {entity_one_hit / len(graph_data)}")


def eval_tris_berts(tris, berts, max_evidence):
    total = 0
    bert_one_hit = 0
    overlap_not_hit = 0
    tris_total = 0
    tris_hit = 0
    bert_total = 0
    bert_hit = 0
    overlap_hit = 0
    overlap_total =0
    empty_tri_s = 0
    not_overlap_hit = 0
    hit_idx = []
    not_overlap_total = 0
    merged = []
    merge_one_hit = 0
    one_doc_hit = 0
    total_docs = 0
    for idx, example in enumerate(berts):
        if example["label"].upper() != "NOT ENOUGH INFO":
            total += 1.0
            all_evi = [[e[2], e[3]] for eg in example["evidence"] for e in eg if e[3] is not None]
            all_evi_docs = list(set([i[0] for i in all_evi]))
            tri_s_l = tris[idx]['triple_sentences']
            sids = [sid for tri in tri_s_l.values() for sid in tri]
            sids = list(set(sids))
            bert_sids = example["predicted_evidence"] if max_evidence is None else \
                    example["predicted_evidence"][:max_evidence]
            tri_sids = [[i.split(c_scorer.SENT_LINE2)[0], int(i.split(c_scorer.SENT_LINE2)[1])] for i in sids]
            if len(sids) == 0:
                empty_tri_s += 1

            overlap_sids = []
            for i in tri_sids:
                if i in bert_sids:
                    overlap_sids.append(i)
            for prediction in overlap_sids:
                if prediction in all_evi:
                    overlap_hit += 1.0
                    break
            overlap_total += len(overlap_sids)

            for prediction in overlap_sids:
                if prediction not in all_evi:
                    overlap_not_hit += 1.0
                    break

            for prediction in bert_sids:
                if prediction in all_evi:
                    bert_one_hit += 1.0
                    break

            for idx, i in enumerate(bert_sids):
                if i in all_evi:
                    bert_hit += 1.0
                    hit_idx.append(idx)
                    if idx > max_evidence:
                        print(idx)

            merged = copy.deepcopy(bert_sids)
            for i in tri_sids:
                if i not in bert_sids:
                    merged.append(i)
                    not_overlap_total += 1
                if i not in bert_sids and i in all_evi:
                    not_overlap_hit += 1

            for prediction in merged:
                if prediction in all_evi:
                    merge_one_hit += 1.0
                    break

            merge_docs = list(set([i[0] for i in merged]))

            for i in merge_docs:
                if i in all_evi_docs:
                    one_doc_hit += 1
                    break

            bert_total += len(bert_sids)
            for i in tri_sids:
                if i in all_evi:
                    tris_hit += 1.0
            tris_total += len(tri_sids)
            example["predicted_evidence"] = bert_sids
    print(f"empty tri_s: {empty_tri_s}")
    print(f"merged at least one doc hit: {one_doc_hit}, total: {total}, rate: {one_doc_hit / total}")
    print(f"merged: at least one ss hit: {merge_one_hit}, total: {total}, rate: {merge_one_hit / total}")
    print(f"overlap hit: {overlap_hit}, overlap total: {overlap_total}, rate: {overlap_hit / overlap_total}")
    print(f"overlap not hit: {overlap_not_hit}, total: {total}, rate: {overlap_not_hit / total}")
    print(f"not overlap hit: {not_overlap_hit}, not overlap total: {not_overlap_total}, rate: {not_overlap_hit / not_overlap_total}")
    print(f"bert hit: {bert_hit}, bert total: {bert_total}, rate: {bert_hit / bert_total}")
    print(f"tri hit: {tris_hit}, tri total: {tris_total}, rate: {tris_hit / tris_total}")
    c_scorer.get_macro_ss_recall_precision(berts, 10)
    count = Counter()
    count.update(hit_idx)
    print(count.most_common())
    # print(sorted(list(count.most_common()), key=lambda x: -x[0]))



def redo_example_docs(data, log_filename):
    for example in tqdm(data):
        es_doc_and_lines, entities, nouns = prepare_candidate_es_for_example(example)
        entity_docs = get_es_entity_links(es_doc_and_lines)
        graph_data_example = prepare_claim_graph_for_example(example, extend_entity_docs=entity_docs, entities=entities, nouns=nouns)
        ent_resource_docs = prepare_candidate2_example(graph_data_example)
        merged = merge_es_and_entity_docs(es_doc_and_lines, ent_resource_docs)
        example['candidate_docs'] = merged
        example['predicted_docids'] = [j.get('id') for j in merged][:10]
    eval_doc_preds(data, 10, log_filename)


if __name__ == '__main__':
    data = read_json_rows(config.RESULT_PATH /"hardset2021/candidate_docs.log")
    redo_example_docs(data, config.RESULT_PATH / "tmp.log")

    # folder = config.RESULT_PATH / "hardset2021"
    # hardset_original = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    # candidate_docs = read_json_rows(folder / "candidate_docs.jsonl")
    # prepare_candidate_sents2_bert_dev(hardset_original, candidate_docs, folder)

    # graph_data = read_json_rows(folder / "claim_graph.jsonl")
    # resource2docs_data = read_json_rows(folder / "graph_resource_docs.jsonl")
    # prepare_candidate_sents3_from_triples(graph_data, resource2docs_data, folder / "tri_ss.jsonl", folder / "tri_ss.log")

    # tri_ss_data = read_json_rows(folder / "tri_ss.jsonl")
    # bert_ss_data = read_json_rows(folder / "bert_ss_0.4_10.jsonl")

    # hit_eval(bert_ss_data, 10)
    # eval_tri_ss(hardset_original, tri_ss_data)
    # eval_tris_berts(tri_ss_data, bert_ss_data, 10)
    # c_scorer.fever_score(bert_ss_data, hardset_original, max_evidence=5, mode={'check_sent_id_correct': True, 'standard': False}, error_analysis_file=folder / "test.log")
    # generate_candidate_graphs(graph_data, tri_ss_data, bert_ss_data,
    #                           folder / "sids.jsonl", folder / "sid2graph.jsonl",
    #                           folder / "sids.log", folder / "sid2graph.log")
    #
    sid2sids_data = read_json_rows(folder / "sids.jsonl")
    docs_data = read_json_rows(folder/ "es_doc_10.jsonl")
