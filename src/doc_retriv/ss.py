from ES.es_search import  search_doc_id_and_keywords_in_sentences
# from utils.c_scorer import *
from utils.fever_db import get_evidence
# from dbpedia_sampler.sentence_util import get_phrases_and_nouns_merged
from dbpedia_sampler.dbpedia_subgraph import construct_subgraph_for_candidate2
from utils.file_loader import save_and_append_results
from doc_retriv.SentenceEvidence import *
from utils.check_sentences import Evidences, sids_to_tuples
import copy
from typing import List
import itertools
from BERT_test.ss_eval import *
from doc_retriv.doc_retrieve_extend import search_entity_docs_for_triples
from bert_serving.client import BertClient
from dbpedia_sampler.uri_util import uri_short_extract2, isURI
from utils.resource_manager import *
from collections import Counter


def filter_bert_claim_vs_sents(claim, docs):
    pass


def prepare_candidate_sents2_bert_dev(original_data, data_with_candidate_docs, output_folder):
    paras = bert_para.PipelineParas()
    paras.pred = True
    paras.mode = 'eval'
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_202101_93.96"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_202101_93.96"
    paras.output_folder = output_folder
    paras.original_data = original_data
    paras.upstream_data = data_with_candidate_docs
    paras.sample_n = 10
    paras.top_n = [10, 5]
    paras.prob_thresholds = [0.4, 0.5]
    pred_ss_and_save(paras)


def prepare_candidate_sents3_from_triples(data_with_graph, data_with_res_doc, output_file, log_file):
    result = []
    with tqdm(total=len(data_with_graph), desc=f"searching triple sentences") as pbar:
        for idx, example in enumerate(data_with_graph):
            claim_dict = example['claim_dict']
            triple_l = claim_dict['graph']
            resouce_doc_dict = data_with_res_doc[idx]['resource_docs']
            triples = []
            for idx_tri, tri in enumerate(triple_l):
                tri['tri_id'] = idx_tri
                triples.append(Triple(tri))
            tri_sentence_dict = search_triples_in_docs(triples, resouce_doc_dict)
            result.append({'id': example['id'], 'triple_sentences': tri_sentence_dict, "triples": [t.__dict__ for t in triples]})
            pbar.update(1)
    save_intermidiate_results(result, output_file)


def get_docid_to_sids_hlinks(all_sids):
    docid2sids = dict()
    sid2hlinkdocs = dict()
    all_doc_ln = [(i.split(c_scorer.SENT_LINE2)[0], int(i.split(c_scorer.SENT_LINE2)[1])) for i in all_sids]
    all_docs = list(set([i[0] for i in all_doc_ln]))
    for doc_id, ln in all_doc_ln:
        _, hlinks = get_text_and_hlinks(doc_id, ln)
        if doc_id in docid2sids:
            docid2sids[doc_id].append(f"{doc_id}{c_scorer.SENT_LINE2}{ln}")
        else:
            docid2sids.update({doc_id: [f"{doc_id}{c_scorer.SENT_LINE2}{ln}"]})
        if len(hlinks) > 0:
            hlinks_lower = [s.lower() for s in hlinks]
            tmp_sid = f"{doc_id}{c_scorer.SENT_LINE2}{ln}"
            for dd in all_docs:
                clean_dd = convert_brc(dd).replace("_", " ").lower()
                if len(list(filter(lambda x: clean_dd == x, hlinks_lower))) > 0:
                    if tmp_sid in sid2hlinkdocs:
                        sid2hlinkdocs[tmp_sid].append(dd)
                    else:
                        sid2hlinkdocs.update({tmp_sid: [dd]})
    sid2hlinksids = dict()
    for s in sid2hlinkdocs:
        linked_docs = sid2hlinkdocs[s]
        linked_s = []
        for d in linked_docs:
            linked_s.extend(docid2sids[d])
        sid2hlinksids[s] = linked_s

    return docid2sids, sid2hlinksids


def prepare_evidence_set_for_bert_nli(data_origin, data_with_bert_s,
                                      data_with_tri_s, data_with_context_graph,
                                      data_sid2sids, data_resource_docs, output_file):
    def get_bert_sids(scored_sentids, threshold=0.5):
        sids = []
        for i in scored_sentids:
            raw_sid = i[0]
            # score = i[-1]
            sid = raw_sid.replace(c_scorer.SENT_LINE, c_scorer.SENT_LINE2)
            sids.append(sid)
        return sids

    def fill_relative_hash(relative_hash, graph: List[Triple]):
        for i in graph:
            if len(i.relatives) > 1:
                relatives = i.relatives
                for r in relatives:
                    if r in relative_hash:
                        relative_hash[r].extend(relatives)
                        relative_hash[r] = list(set(relative_hash[r]))
                        relative_hash[r].remove(r)
        for i in relative_hash:
            if len(relative_hash[i]) == 0:
                for tri in graph:
                    if len(list(filter(lambda x: i in x or x in i, tri.keywords))) > 0:
                        relative_hash[i].append(tri.text)


    def init_relative_hash(relative_hash=None):
        if relative_hash is not None and len(relative_hash) != 0:
            relative_hash = {i : [] for i in relative_hash}
        else:
            linked_phrases_l = claim_dict['linked_phrases_l']
            not_linked_phrases_l = claim_dict['not_linked_phrases_l']
            linked_phrases = [i['text'] for i in linked_phrases_l]
            all_phrases = not_linked_phrases_l + linked_phrases
            relative_hash = {key: [] for key in all_phrases}
        return relative_hash

    def is_well_linked(sub_graph, relative_hash=None):
        relative_hash = init_relative_hash(relative_hash)
        fill_relative_hash(relative_hash, sub_graph)
        tmp_no_relatives_found = []
        has_relatives = []
        no_relatives_found = []
        for i in relative_hash:
            if len(relative_hash[i]) == 0:
                tmp_no_relatives_found.append(i)
            else:
                has_relatives.append(i)
        for i in tmp_no_relatives_found:
            if len(list(filter(lambda x: i in x or x in i, has_relatives))) == 0:
                no_relatives_found.append(i)
        if len(no_relatives_found) > 0:
            return False
        else:
            return True

    def get_subgraph_docids(sub_graph):
        subgraph_resources = []
        for tri in sub_graph:
            if tri.subject not in subgraph_resources:
                subgraph_resources.append(tri.subject)
            if isURI(tri.object) and tri.object not in subgraph_resources:
                subgraph_resources.append(tri.object)
        subgraph_docids = []
        for res in subgraph_resources:
            if res in resource2docids:
                res_docids = resource2docids[res]
                if len(res_docids) > 0:
                    subgraph_docids.extend([r['id'] for r in res_docids])
        return subgraph_docids

    def add_linked_doc_ss(evi_set):
        add_linked_sid = []
        for e_s in evi_set:
            if len(e_s) < 3:
                e_sids = e_s.to_sids()
                for ss in e_sids:
                    if ss in sid2linkedsids and len(sid2linkedsids[ss]) > 0:
                        linked_sids = sid2linkedsids[ss]
                        for l_s in linked_sids:
                            tmp_e_s = copy.deepcopy(e_s)
                            tmp_e_s.add_sent_sid(l_s)
                            add_linked_sid.append(tmp_e_s)
        return add_linked_sid



    for idx, example in enumerate(data_origin):
        if idx < 3:
            continue
        # ["Soul_Food_-LRB-film-RRB-<SENT_LINE>0", 1.4724552631378174, 0.9771634340286255]
        bert_s = get_bert_sids(data_with_bert_s[idx]['scored_sentids'])
        triples = [Triple(t_dict) for t_dict in data_with_tri_s[idx]['triples']]
        claim_dict = data_with_context_graph[idx]['claim_dict']
        linked_phrases = claim_dict['linked_phrases_l']
        linked_entities = [ent['text'] for ent in linked_phrases]
        sid2sids = data_sid2sids[idx]['sid2sids']
        resource2docids = data_resource_docs[idx]['resource_docs']
        candidate_sid_sets = []
        tri_s = list(set([s for tt in triples for s in tt.sentences]))
        bert_and_tri_s = list(set(bert_s) | set(tri_s))
        docid2sids, sid2linkedsids = get_docid_to_sids_hlinks(bert_and_tri_s)

        for linked_ent in linked_entities:
            for docid in docid2sids:
                if linked_ent.lower() in convert_brc(docid).replace("_", " ").lower():
                    tmp_sids = docid2sids[docid]
                    sid_combination = generate_sentence_combination(tmp_sids)
                    candidate_sid_sets.extend(sid_combination)

        candidate_sid_sets.extend(add_linked_doc_ss(candidate_sid_sets))

        if len(triples) > 0:
            subgraphs, _ = generate_triple_subgraphs(triples, claim_dict)
            claim_relative_hash = init_relative_hash()
            well_linked_sg_idx = []
            partial_linked_idx = []
            all_subgraph_sids = []
            for subgraph in subgraphs:
                tmp_sids = []
                for tt in subgraph:
                    tmp_sids.append(tt.sentences)
                all_subgraph_sids.append(tmp_sids)
            for idx, subgraph in enumerate(subgraphs):
                subgraph_sids = all_subgraph_sids[idx]
                if is_well_linked(subgraph, claim_relative_hash) and all([len(s) > 0 for s in subgraph_sids]):
                    well_linked_sg_idx.append(idx)
                else:
                    partial_linked_idx.append(idx)
            if len(well_linked_sg_idx) > 0:
                for idx in well_linked_sg_idx:
                    good_subgraph = subgraphs[idx]
                    tmp_sid_sets = generate_triple_sentence_combination(good_subgraph, [])
                    candidate_sid_sets.extend(tmp_sid_sets)
            elif len(partial_linked_idx) > 0:
                for idx in partial_linked_idx:
                    subgraph_sids = all_subgraph_sids[idx]
                    if all([len(s) == 0 for s in subgraph_sids]):
                        continue
                    else:
                        tmp_sid_sets = generate_triple_sentence_combination(subgraph, [])
                        tmp_sid_sets.extend(add_linked_doc_ss(tmp_sid_sets))
                        candidate_sid_sets.extend(tmp_sid_sets)
        else:
            candidate_sid_sets = generate_sentence_combination(bert_s)
        candidate_sid_sets = list(set(candidate_sid_sets))
        example.update({'nli_sids': candidate_sid_sets})
    save_intermidiate_results(data_origin, output_file)


# the resource_to_doc_dict is constructed in advance for performance concern
def search_triples_in_docs(triples: List[Triple], docs:dict):  #  list[Triple]
    # phrase match via ES
    # possible_sentences = []
    for tri in triples:
        resource_docs = []
        if len(docs) > 0:
            if tri.subject in docs:
                resource_docs.extend(docs[tri.subject])
            if tri.object in docs:
                resource_docs.extend(docs[tri.object])
        if len(resource_docs) > 0:
            for doc in resource_docs:
                doc_id = doc['id']
                subject_text = uri_short_extract2(tri.subject)
                tmp_sentences = search_doc_id_and_keywords_in_sentences(doc_id, subject_text, tri.keywords)
                if len(tmp_sentences) > 0:
                    for i in tmp_sentences:
                        # i['tri_id'] = tri.tri_id
                        tri.sentences.append(i['sid'])
                    # sentences.extend([SentenceEvidence(ts) for ts in tmp_sentences])
        # if len(sentences) > 0:
        #     possible_sentences.extend(sentences)
    tri_sentence_dict = {tri.tri_id: list(set(tri.sentences)) for tri in triples}
    return tri_sentence_dict


def search_sentences_for_triples(triples: List[Triple]):
    # phrase match via ES
    # possible_sentences = []
    resource_to_entity_pages = search_entity_docs_for_triples(triples)
    return search_triples_in_docs(triples, resource_to_entity_pages)


# def search_triple_in_sentence(tri, doc_id):
#     # phrase match via ES
#     possible_sentences = []
#     sentences = []
#     tmp_sentences = search_doc_id_and_keywords_in_sentences(doc_id, tri['keywords'])
#     if len(tmp_sentences) > 0:
#         for i in tmp_sentences:
#             i['tri_id'] = tri.tri_id
#             tri.sentences.add(i['sid'])
#         sentences.extend([SentenceEvidence(ts) for ts in tmp_sentences])
#     if len(sentences) > 0:
#         possible_sentences.extend(sentences)
#     return possible_sentences


def strategy_over_all(data_origin, data_with_bert_s, data_with_tri_s, data_with_context_graph, data_sid2sids, output_file):
    # 1. well linked and tri_s in bert_s -> tri_s ^ bert_S
    # 2. well linked and tri_s not in bert_s -> tri_s + bert_s
    # 3. partially linked and tri_s > 0 -> tri_s + bert_s + extend
    # 4. not linked or tri_s == 0 -> bert_s
    # with tqdm(total=len(data_origin), desc=f"searching triple sentences") as pbar:
    #     for idx, example in enumerate(data_origin):
    #         bert_s = get_bert_sids(data_with_bert_s[idx]['scored_sentids'])
    #         triples = [Triple(t_dict) for t_dict in data_with_tri_s[idx]['triples']]
    #         context_graph = data_with_context_graph[idx]['claim_dict']
    #         sid2sids = data_sid2sids[idx]['sid2sids']
    #         ents = get_linked_entities(context_graph)
    pass



def merge_sentences_and_generate_evidence_set(linked_triples_with_sentences: List[Triple],
                                              candidate_sentences: List[str],
                                              claim_dict, sid2sids):
    subgraphs, subgraph_sentences_l = generate_triple_subgraphs(linked_triples_with_sentences, claim_dict)
    evidence_set_from_triple = []
    for sub_g in subgraphs:
        tmp_evi_from_tri_subgraph = generate_triple_sentence_combination(sub_g, [])
        evidence_set_from_triple.extend(tmp_evi_from_tri_subgraph)
    evidence_set_from_triple = list(set(evidence_set_from_triple))
    evidence_set_from_sentences = generate_sentence_combination(candidate_sentences)
    new_evidence_set = copy.deepcopy(evidence_set_from_triple)
    for evid_s in evidence_set_from_sentences:
        for evid_t in evidence_set_from_triple:
            new_evidence = copy.deepcopy(evid_t)
            new_evidence.add_sents_tuples(evid_s.evidences_list)
            new_evidence_set.append(new_evidence)
        new_evidence_set.append(evid_s)
    new_evidence_set = list(set(new_evidence_set))
    extend_evidences = []
    for e_s in new_evidence_set:
        for doc, ln in e_s.evidences_list:
            e_sid = f"{doc}{c_scorer.SENT_LINE2}{ln}"
            if e_sid in sid2sids:
                extend_sids = sid2sids[e_sid]
                for extend_s in extend_sids:
                    new_evidence = copy.deepcopy(e_s)
                    new_evidence.add_sent_sid(extend_s)
                    extend_evidences.append(new_evidence)
    if len(extend_evidences) > 0:
        new_evidence_set.extend(extend_evidences)
    new_evidence_set = list(set(new_evidence_set))
    return new_evidence_set

def merge_sentences_and_generate_evidence_set2(subgraph: List[Triple],
                                              candidate_sentences: List[str]):
    evidence_set_from_triple = generate_triple_sentence_combination(subgraph, [])
    evidence_set_from_triple = list(set(evidence_set_from_triple))
    evidence_set_from_sentences = generate_sentence_combination(candidate_sentences)
    new_evidence_set = copy.deepcopy(evidence_set_from_triple)
    for evid_s in evidence_set_from_sentences:
        for evid_t in evidence_set_from_triple:
            if len(evid_s) + len(evid_t) < 4:
                new_evidence = copy.deepcopy(evid_t)
                new_evidence.add_sents_tuples(evid_s.evidences_list)
                new_evidence_set.append(new_evidence)
        new_evidence_set.append(evid_s)
    new_evidence_set = list(set(new_evidence_set))
    return new_evidence_set


def generate_sentence_combination(list_of_sentences: List):
    new_evidence_set = set()
    max_s = 3 if len(list_of_sentences) > 3 else len(list_of_sentences)
    for i in reversed(range(1, max_s + 1)):
        combination_set = itertools.combinations(list_of_sentences, i)
        for c in combination_set:
            sids = [s for s in c]
            raw_doc_ln = sids_to_tuples(sids)
            evid = Evidences(raw_doc_ln)
            new_evidence_set.add(evid)
    return list(new_evidence_set)


def generate_expand_evidence_from_hlinks(possible_evidence, hlink_docs):
    return []


# each sentence may have different subgraphs, as one entity may linked to different resources
def generate_triple_subgraphs(list_of_triples: List[Triple], claim_dict):
    def has_overlap_resource(tri_l1_tomerge, tri_l2_desc, tri_dict):
        for tri1_id in tri_l1_tomerge:
            tri1_res_obj = tri_dict[tri1_id].object
            tri1_res_subj = tri_dict[tri1_id].subject
            tri1_text = tri_dict[tri1_id].text
            for tri2_id in tri_l2_desc:
                tri2_res_obj = tri_dict[tri2_id].object
                tri2_res_subj = tri_dict[tri2_id].subject
                tri2_text = tri_dict[tri2_id].text
                if tri1_text == tri2_text \
                        and (tri1_res_obj == tri2_res_subj
                             or tri1_res_subj == tri2_res_obj):
                    return True
                if tri1_text != tri2_text \
                        and (tri1_res_obj == tri2_res_subj
                             or tri1_res_subj == tri2_res_obj
                             or tri2_res_obj == tri1_res_obj):
                    return True
        return False

    def extend_tri_set(tri_l1_tomerge, tri_l2_desc, tri_dict):
        relative2 = list(set([r for t2 in tri_l2_desc for r in tri_dict[t2].relatives]))
        desc_keywords = list(set([k for x in tri_l2_desc for k in tri_dict[x].keywords]))
        desc_rel_and_keywords = list(set(relative2) | set(desc_keywords))
        for t1 in tri_l1_tomerge:
            t1_rel = tri_dict[t1].relatives
            if any([len(list(filter(lambda x: k1 in x or x in k1, desc_rel_and_keywords))) == 0 for k1 in t1_rel]):
                tri_l2_desc.append(t1)
        return tri_l2_desc

    def test(tri_l, tri_dict, linked_l):
        if len(tri_l) == 0:
            return linked_l
        else:
            r_t = tri_l.pop()
            if len(linked_l) == 0:
                linked_l.append(r_t)
            else:
                for linked_t in linked_l:
                    if has_overlap_resource(r_t, linked_t, tri_dict):
                        dup_linked_t2 = copy.deepcopy(linked_t)
                        extend_tri_set(r_t, dup_linked_t2, tri_dict)
                        if dup_linked_t2 not in linked_l:
                            linked_l.append(dup_linked_t2)
                    if has_overlap_resource(linked_t, r_t, tri_dict):
                        dup_r_t = copy.deepcopy(r_t)
                        extend_tri_set(linked_t, dup_r_t, tri_dict)
                        if dup_r_t not in linked_l:
                            linked_l.append(dup_r_t)
                    if r_t not in linked_l:
                        linked_l.append(r_t)
            return test(tri_l, tri_dict, linked_l)



    # def can_extend_each_other(tri_l1_tomerge, tri_l2_desc, tri_dict):
    #     text1 = list(set([tri_dict[t1].text for t1 in tri_l1_tomerge]))
    #     text2 = list(set([tri_dict[t2].text for t2 in tri_l2_desc]))
    #     if all([x in text2 for x in text1]):
    #         return False
    #     relative1 = list(set([r for t1 in tri_l1_tomerge for r in tri_dict[t1].relatives]))
    #     relative2 = list(set([r for t2 in tri_l2_desc for r in tri_dict[t2].relatives]))
    #     src_keywords = list(set([k for x in tri_l1_tomerge for k in tri_dict[x].keywords]))
    #     desc_keywords = list(set([k for x in tri_l2_desc for k in tri_dict[x].keywords]))
    #     # to_merge is better linked than desc, we will keep to_merge alone and  no need to extend it to desc
    #     l1_is_better = True
    #     for r2 in relative2:
    #         # there exists item from l2 that not exist in l1
    #         if len(list(filter(lambda x: r2 in x or x in r2, relative1 + src_keywords))) == 0:
    #             l1_is_better = False
    #     if l1_is_better:
    #         return False
    #
    #     can_extend = False
    #     for r1 in relative1:
    #         if len(list(filter(lambda x: r1 in x or x in r1, relative2))) == 0 \
    #                 and len(list(filter(lambda x: r1 in x or x in r1, desc_keywords))) == 0:
    #             can_extend = True
    #             break
    #     return can_extend

    tri_id_to_tri_dict = {t.tri_id: t for t in list_of_triples}
    resource_to_phrase_dict = dict()
    linked_phrases = claim_dict['linked_phrases_l']
    for lp in linked_phrases:
        text = lp['text']
        links = lp['links']
        for l in links:
            res = l['URI']
            if res not in resource_to_phrase_dict:
                resource_to_phrase_dict.update({res: text})

    resource_to_tris_dict = dict()
    for tri in list_of_triples:
        # if len(tri.sentences) == 0:
        #     continue
        subj = tri.subject
        # obj = tri.object
        if subj in resource_to_tris_dict:
            resource_to_tris_dict[subj].append(tri.tri_id)
        else:
            resource_to_tris_dict.update({subj: [tri.tri_id]})

    all_resource_tri_sets = list(resource_to_tris_dict.values())
    subgraphs = test(all_resource_tri_sets, tri_id_to_tri_dict, [])

    # ph_to_trisets_dict = dict()
    # for res in resource_to_tris_dict:
    #     ph = resource_to_phrase_dict[res]
    #     tri_ids = resource_to_tris_dict[res]
    #     if ph in ph_to_trisets_dict:
    #         ph_to_trisets_dict[ph].append(tri_ids)
    #     else:
    #         ph_to_trisets_dict.update({ph: [tri_ids]})

    # for ph in ph_to_trisets_dict:
    #     ph_tri_subsets = copy.deepcopy(ph_to_trisets_dict[ph])
    #     if len(subgraphs) == 0:
    #         subgraphs = ph_tri_subsets
    #     else:
    #         for idx, ph_tri in enumerate(ph_tri_subsets):
    #             if
    #         for tri_ids_to_merge in ph_tri_subsets:
    #             # has_overlap = False
    #             for single_tri_set in subgraphs:
    #                 if has_overlap_resource(tri_ids_to_merge, single_tri_set, tri_id_to_tri_dict):
    #                     extend_tri_set(tri_ids_to_merge, single_tri_set)
    #                     # has_overlap = True
    #             # if not has_overlap:
    #             #     for single_tri_set in subgraphs:
    #             #         if can_extend_each_other(tri_ids_to_merge, single_tri_set, tri_id_to_tri_dict):
    #             #             new_tri_sets = copy.deepcopy(single_tri_set)
    #             #             new_tri_sets.extend(tri_ids_to_merge)
    #             #             subgraphs.append(new_tri_sets)
    #                 subgraphs.append(tri_ids_to_merge)

    subgraphs = [list(set(s)) for s in subgraphs]
    tri_sets_with_triples = []
    for s in subgraphs:
        tmp_set = [tri_id_to_tri_dict[id] for id in s]
        tri_sets_with_triples.append(tmp_set)

    sentences_l = []
    for single_sub_g in tri_sets_with_triples:
        single_sub_g_sents = []
        for tri in single_sub_g:
            single_sub_g_sents.extend(tri.sentences)
        sentences_l.append(list(set(single_sub_g_sents)))

    # graph_sentences = graph_to_sentences(tri_sets_with_triples)
    # filtered_senteces = []
    # filtered_subgraphs = []
    # for idx, sentence_list in enumerate(graph_sentences):
    #     if len(sentence_list) != 0:
    #         filtered_senteces.append(sentence_list)
    #         filtered_subgraphs.append(tri_sets_with_triples[idx])
    # return filtered_subgraphs, filtered_senteces
    return tri_sets_with_triples, sentences_l


def graph_to_sentences(tri_sets):
    sentences_l = []
    for subgraph in tri_sets:
        graph_sentences = []
        for tri in subgraph:
            sentences = tri.sentences
            for s in sentences:
                if s not in graph_sentences:
                    graph_sentences.append(s)
        sentences_l.append(graph_sentences)
    return sentences_l


def generate_triple_sentence_combination(list_of_triples: List[Triple], list_of_evidence: List[Evidences]):
    if len(list_of_triples) == 0:
        return list_of_evidence
    else:
        tmp_triples = copy.deepcopy(list_of_triples)
        triple = tmp_triples.pop()
        new_evidence_l = list_of_evidence
        if len(list_of_evidence) == 0:
            raw_doc_ln = sids_to_tuples(triple.sentences)
            new_evidence_l = [Evidences([doc_ln]) for doc_ln in raw_doc_ln]
        else:
            for tri_sid in triple.sentences:
                tmp_evidence_l = copy.deepcopy(list_of_evidence)
                for e in tmp_evidence_l:
                    e.add_sent_sid(tri_sid)
                new_evidence_l.extend(tmp_evidence_l)
                new_evidence_l = list(set(new_evidence_l))
        return generate_triple_sentence_combination(tmp_triples, new_evidence_l)


def update_relative_hash(relative_hash:dict, connected_phrases):
    for i in connected_phrases:
        if i in relative_hash:
            relative_hash[i].extend(connected_phrases)
            relative_hash[i].remove(i)
            relative_hash[i] = list(set(relative_hash[i]))


def find_isolated_entity(subgraph: List[Triple], graph_context):
    linked_phrases_l = graph_context['linked_phrases_l']
    isolated_entity = []
    all_relatives = []
    for tri in subgraph:
        all_relatives.extend(tri.relatives)
    all_relatives = list(set(all_relatives))
    for i in linked_phrases_l:
        entity_ph = i['text']
        if entity_ph not in all_relatives:
            isolated_entity.append(i)
    return isolated_entity


def generate_candidate_graphs(data_with_graph, data_with_tri_s, data_with_s, sid_output_file, graph_output_file, sid_log, graph_log):
    def get_bert_sids(bert_item, tri_item, max_evidence=10):
        bert_sids = bert_item["predicted_sentids"] if max_evidence is None else \
            bert_item["predicted_sentids"][:max_evidence]
        sids = [raw_sid.replace(c_scorer.SENT_LINE, c_scorer.SENT_LINE2) for raw_sid in bert_sids]
        tri_s_l = tri_item['triple_sentences']
        tri_sids = [sid for tri in tri_s_l.values() for sid in tri]
        for ts in tri_sids:
            if ts not in sids:
                sids.append(ts)
        return sids

    sid_to_extend_sids_l = []
    candidate_context_graph_l = []
    batch = 10
    flush_num = batch
    with tqdm(total=len(data_with_graph), desc=f"constructing candidate graphs") as pbar:
        for idx, bert_example in enumerate(data_with_s):
            sids = get_bert_sids(bert_example, data_with_tri_s[idx])
            # triples = [Triple(t_dict) for t_dict in data_with_tri_s[idx]['triples']]
            claim_dict = data_with_graph[idx]['claim_dict']
            sid_to_extend_sids, sid_to_graph = extend_candidate_one_hop(claim_dict, sids)
            sid_to_extend_sids_l.append({'id': bert_example['id'], 'sid2sids': sid_to_extend_sids})
            candidate_context_graph_l.append({'id': bert_example['id'], 'sid2graphs': sid_to_graph})
            flush_num -= 1
            if flush_num == 0 or idx == (len(data_with_s) - 1):
                save_and_append_results(sid_to_extend_sids_l, idx + 1, sid_output_file, sid_log)
                save_and_append_results(candidate_context_graph_l, idx + 1, graph_output_file, graph_log)
                flush_num = batch
                sid_to_extend_sids_l = []
                candidate_context_graph_l = []
            pbar.update(1)


def get_text_and_hlinks(doc_id, ln):
    feverDB = FeverDBResource()
    _, text, hlinks = get_evidence(feverDB.get_cursor(), doc_id, ln)
    hlinks_l = json.loads(hlinks)
    return text, list(set(hlinks_l))


# candidate_sentences: list of sids
def extend_candidate_one_hop(claim_dict, candidate_sentences: List[str]):
    # 5. isolated_nodes candidate docs 2 sentences to sent_context_graph -> new entities
    # 6. ES search sent_context_graph new entity pages -> candidate docs 3
    # 7. BERT filter:  extended context triples VS candidate docs 3 sentences  -> candidate sentence 3
    # 8. aggregate envidence set -> candidate sentences 4
    sid_to_extend_sids = dict()
    sid2graph_l = dict()
    claim_linked_resources = claim_dict["linked_phrases_l"]
    claim_linked_phs = [i['text'] for i in claim_linked_resources]

    for candidate_sent_sid in candidate_sentences:
        # extend_triples is in json format
        doc_title, ln = candidate_sent_sid.split(c_scorer.SENT_LINE2)[0], int(candidate_sent_sid.split(c_scorer.SENT_LINE2)[1])
        candidate_sent_text, hlinks = get_text_and_hlinks(doc_title, ln)
        candidate_context_graph, extend_triples_json = construct_subgraph_for_candidate2(candidate_sent_text,
                                                                                    sid=candidate_sent_sid,
                                                                                    additional_phrase=claim_linked_phs,
                                                                                    additional_resources=claim_linked_resources)

        sid2graph_l.update({candidate_sent_sid: {'graph': candidate_context_graph, 'extend_triples:': extend_triples_json}})

        extend_triples = []
        for idx_tri, tri in enumerate(extend_triples_json):
            tri['tri_id'] = f"{idx_tri}_extend"
            extend_triples.append(Triple(tri))
        extend_triples_dict = search_sentences_for_triples(extend_triples)  # tri.tri_id:list(set(tri.sentences))
        all_e_sids = []
        for ex_sids in extend_triples_dict.values():
            if len(ex_sids) > 0:
                all_e_sids.extend(ex_sids)
        sid_to_extend_sids.update({candidate_sent_sid: list(set(all_e_sids))})
    return sid_to_extend_sids, sid2graph_l


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


def get_hlink_docs(evidence):
    return []

def filter_bert_claim_vs_triplesents_and_hlinks(claim, sentence):
    pass
    # h_links = sentence['h_links']
    # h_links_docs = search_entity_docs(h_links)
    # for doc_id in h_links_docs:
    #     cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
    #     # Merging to data list and removing duplicate
    #     for i in range(len(cur_r_list)):
    #         if cur_id_list[i] in id_list:
    #             continue
    #         else:
    #             r_list.append(cur_r_list[i])
    #             id_list.append(cur_id_list[i])
    #
    # # assert len(id_list) == len(set(id_list))  # check duplicate
    # # assert len(r_list) == len(id_list)
    # if not (len(id_list) == len(set(id_list)) or len(r_list) == len(id_list)):
    #     utils.get_adv_print_func(err_log_f)
    #
    # zipped_s_id_list = list(zip(r_list, id_list))
    # # Sort using id
    # # sorted(evidences_set, key=lambda x: (x[0], x[1]))
    # zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))
    #
    # all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
    #                                           id_tokenized=True)

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


if __name__ == '__main__':
    folder = config.RESULT_PATH / "hardset2021"
    hardset_original = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    # candidate_docs = read_json_rows(folder / "candidate_docs.jsonl")
    # prepare_candidate_sents2_bert_dev(hardset_original, candidate_docs, folder)

    graph_data = read_json_rows(folder / "claim_graph.jsonl")
    resource2docs_data = read_json_rows(folder / "graph_resource_docs.jsonl")
    # prepare_candidate_sents3_from_triples(graph_data, resource2docs_data, folder / "tri_ss.jsonl", folder / "tri_ss.log")

    tri_ss_data = read_json_rows(folder / "tri_ss.jsonl")
    bert_ss_data = read_json_rows(folder / "bert_ss_0.4_10.jsonl")

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
    prepare_evidence_set_for_bert_nli(hardset_original, bert_ss_data, tri_ss_data, graph_data, sid2sids_data, resource2docs_data, folder / "nli_sids.jsonl")


