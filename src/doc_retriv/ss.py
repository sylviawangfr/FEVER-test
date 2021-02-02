from ES.es_search import  search_doc_id_and_keywords_in_sentences
# from utils.c_scorer import *
from utils.fever_db import *
# from dbpedia_sampler.sentence_util import get_phrases_and_nouns_merged
from dbpedia_sampler.dbpedia_subgraph import construct_subgraph_for_candidate2

from doc_retriv.SentenceEvidence import *
from utils.check_sentences import Evidences, sids_to_tuples
import copy
from typing import List
import itertools
from BERT_test.ss_eval import *
from doc_retriv.doc_retrieve_extend import search_entity_docs_for_triples
from bert_serving.client import BertClient
from dbpedia_sampler.uri_util import uri_short_extract2, isURI


def filter_bert_claim_vs_sents(claim, docs):
    pass


def prepare_candidate_sents2_bert_dev(original_data, data_with_candidate_docs, output_folder):
    paras = bert_para.PipelineParas()
    paras.pred = True
    paras.mode = 'eval'
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_202101_93.9"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_202101_93.9"
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


def get_bert_sids(scored_sentids, threshold=0.5):
    sids = []
    for i in scored_sentids:
        raw_sid = i[0]
        # score = i[-1]
        sid = raw_sid.replace(c_scorer.SENT_LINE, c_scorer.SENT_LINE2)
        sids.append(sid)
    return sids


def prepare_evidence_set_for_bert_nli(data_origin, data_with_bert_s, data_with_tri_s, data_with_context_graph, output_file):
    for idx, example in enumerate(data_origin):
        # ["Soul_Food_-LRB-film-RRB-<SENT_LINE>0", 1.4724552631378174, 0.9771634340286255]
        bert_s = get_bert_sids(data_with_bert_s[idx]['scored_sentids'])
        triples = [Triple(t_dict) for t_dict in data_with_tri_s[idx]['triples']]
        context_graph = data_with_context_graph[idx]['claim_dict']
        sid_sets = merge_sentences_and_generate_evidence_set(triples, bert_s, context_graph)
        example.update({'nli_sids': sid_sets})
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
                tmp_sentences = search_doc_id_and_keywords_in_sentences(doc_id, tri.text, tri.keywords)
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


# def strategy_over_all(claim):
#     # 1. ES search phrases
#     nouns = get_phrases_and_nouns_merged(claim)
#     # 2. get ES page candidates -> candidate docs 1
#     #  . BERT filter: claim VS candidate docs 1 sents -> candidate sentences 1
#     candidate_docs_1 = search_and_merge2(nouns)
#     # candidate_sentences_1 = filter_bert_claim_vs_sents(claim, candidate_docs_1)
#     claim_dict = construct_subgraph_for_claim(claim)
#     claim_graph = claim_dict['graph']
#     if len(claim_graph) > 0:
#         # 1. ES search all linked entity page -> candidate docs 2
#         candidate_docs_2 = search_entity_docs_for_triples(claim_graph)
#         candidate_docs = candidate_docs_1 + candidate_docs_2
#         # 2. BERT filter: claim VS candidate docs sents -> candidate sentences 2
#         candidate_sents_2 = filter_bert_claim_vs_sents(claim, candidate_docs)
#         linked_triples = get_linked_triples(claim_graph)
#         candidate_sentences_3 = []
#         if len(linked_triples) > 0:
#             # 3. ES filter: linked triples VS candidate docs 2 -> candidate sentences 3
#             candidate_sentences_3 = search_triples_in_docs(linked_triples, candidate_docs_2)
#         # 4. sort candidate sentences 2 + 3 -> candidate sentence 4
#         candidate_sentences_4 = candidate_sents_2 + candidate_sentences_3
#         isolated_phrases = claim_dict['no_relatives']
#         if len(isolated_phrases) > 0:
#             # 5. isolated_nodes candidate docs 2 sentences to sent_context_graph -> new entities
#             # 6. ES search sent_context_graph new entity pages -> candidate docs 3
#             # 7. BERT filter:  extended context triples VS candidate docs 3 sentences  -> candidate sentence 3
#             # 8. aggregate envidence set -> candidate sentences 4
#             candidate_sentence_set = strategy_one_hop(claim_dict, linked_triples, candidate_sentences_4)
#
#     else:
#         # *. BERT filter: claim VS candidate docs 1 sents -> candidate sentences 1
#         # cannot extract context graph, return candidate sentences 1
#         candidate_docs = candidate_docs_1
#         candidate_sentences_1 = filter_bert_claim_vs_sents(claim, candidate_docs_1)
#     pass



def merge_sentences_and_generate_evidence_set(linked_triples_with_sentences: List[Triple],
                                              candidate_sentences: List[str],
                                              context_graph):
    subgraphs, subgraph_sentences_l = generate_triple_subgraphs(linked_triples_with_sentences, context_graph)
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
    # for e_s in new_evidence_set:
    #     for s in e_s.evidences_list:
    #         extend_sentences = candidate_sentences[s].extend_sentences
    #         for extend_s in extend_sentences:
    #             new_evidence = copy.deepcopy(e_s)
    #             new_evidence.add_sent_sid(extend_s)
    #             new_evidence_set.append(new_evidence)
    # new_evidence_set = list(set(new_evidence_set))
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
def generate_triple_subgraphs(list_of_triples: List[Triple], context_graph):
    def has_overlap_resource(tri_l1, tri_l2, tri_dict):
        for tri1_id in tri_l1:
            tri1_res = {tri_dict[tri1_id].object, tri_dict[tri1_id].subject}
            for tri2_id in tri_l2:
                tri2_res = {tri_dict[tri2_id].object, tri_dict[tri2_id].subject}
                if len(tri1_res & tri2_res) > 0:
                    return True
        return False

    tri_id_to_tri_dict = {t.tri_id: t for t in list_of_triples}
    resource_to_phrase_dict = dict()
    linked_phrases = context_graph['linked_phrases_l']
    for lp in linked_phrases:
        text = lp['text']
        links = lp['links']
        for l in links:
            res = l['URI']
            if res not in resource_to_phrase_dict:
                resource_to_phrase_dict.update({res: text})

    resource_to_tris_dict = dict()
    for tri in list_of_triples:
        if len(tri.sentences) == 0:
            continue
        subj = tri.subject
        obj = tri.object
        if subj in resource_to_tris_dict:
            resource_to_tris_dict[subj].append(tri.tri_id)
        else:
            resource_to_tris_dict.update({subj: [tri.tri_id]})

        if obj in resource_to_phrase_dict:
            if obj in resource_to_tris_dict:
                resource_to_tris_dict[obj].append(tri.tri_id)
            else:
                resource_to_tris_dict.update({obj: [tri.tri_id]})

    ph_to_trisets_dict = dict()
    for res in resource_to_tris_dict:
        ph = resource_to_phrase_dict[res]
        tri_ids = resource_to_tris_dict[res]
        if ph in ph_to_trisets_dict:
            ph_to_trisets_dict[ph].append(tri_ids)
        else:
            ph_to_trisets_dict.update({ph: [tri_ids]})

    subgraphs = []
    for ph in ph_to_trisets_dict:
        tri_subsets = ph_to_trisets_dict[ph]
        if len(subgraphs) == 0:
            subgraphs = tri_subsets
        else:
            for tri_ids_to_merge in tri_subsets:
                need_expand = True
                for single_tri_set in subgraphs:
                    if has_overlap_resource(single_tri_set, tri_ids_to_merge, tri_id_to_tri_dict):
                        single_tri_set.extend(tri_ids_to_merge)
                        need_expand = False
                        break
                if need_expand:
                    new_tri_sets = copy.deepcopy(subgraphs)
                    for new_tri_s in new_tri_sets:
                        new_tri_s.extend(tri_ids_to_merge)
                        subgraphs.append(new_tri_s)

    subgraphs = [list(set(s)) for s in subgraphs]
    merged = []
    for s in subgraphs:
        if s not in merged:
            merged.append(s)
    tri_sets_with_triples = []
    for s in merged:
        tmp_set = [tri_id_to_tri_dict[id] for id in s]
        tri_sets_with_triples.append(tmp_set)

    graph_sentences = graph_to_sentences(tri_sets_with_triples)
    filtered_senteces = []
    filtered_subgraph = []
    for idx, sentence_list in enumerate(graph_sentences):
        if len(sentence_list) != 0:
            filtered_senteces.append(sentence_list)
            filtered_subgraph.append(tri_sets_with_triples[idx])
    return filtered_subgraph, filtered_senteces


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
        triple = list_of_triples.pop()
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
        return generate_triple_sentence_combination(list_of_triples, new_evidence_l)


def get_linked_triples(context_graph):
    tris = []
    for t in context_graph:
        if t['relation'] != '' and t['object'] != '':
            tris.append(Triple(t))
    return tris


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


def generate_extend_sentences(data_with_graph, data_with_tri_s, data_with_s, output_file):
    bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    for idx, example in enumerate(data_with_s):
        scored_sentids = example['scored_sentids']
        sids = get_bert_sids(scored_sentids)
        triples = [Triple(t_dict) for t_dict in data_with_tri_s[idx]['triples']]
        claim_dict = data_with_graph[idx]['claim_dict']
        # isolated_nodes = data_with_graph[idx]['isolated_nodes']
        no_relative_phrase = claim_dict['no_relatives']
        linked_pharses = {i['text']: i for i in claim_dict['linked_phrases_l']}
        isolated_nodes = []
        for i in no_relative_phrase:
            if i in linked_pharses:
                isolated_nodes.append(linked_pharses[i])
        # 1. more than one entity is not linked
        if len(isolated_nodes) > 0:
            sid_to_extend_sids, candidate_context_graph, extend_triples_json = strategy_one_hop(claim_dict, triples, sids, bc=bc)



# candidate_sentences: list of sids
def strategy_one_hop(claim_dict, subgraph: List[Triple], candidate_sentences: List[str], bc:BertClient=None):
    # 5. isolated_nodes candidate docs 2 sentences to sent_context_graph -> new entities
    # 6. ES search sent_context_graph new entity pages -> candidate docs 3
    # 7. BERT filter:  extended context triples VS candidate docs 3 sentences  -> candidate sentence 3
    # 8. aggregate envidence set -> candidate sentences 4
    # extend_evidence_l = []
    # isolated_nodes_copy = copy.deepcopy(isolated_nodes)
    doc_and_lines = dict()
    for s in candidate_sentences:
        doc_id, ln = s.split(c_scorer.SENT_LINE2)
        if doc_id in doc_and_lines:
            doc_and_lines[doc_id].append(ln)
        else:
            doc_and_lines.update({doc_id: [ln]})

    tri_sentences = []
    has_relatives = []

    resources_in_graph = []
    for tri in subgraph:
        resources_in_graph.append(tri.subject)
        if tri.object != '':
            resources_in_graph.append(tri.object)
        if len(tri.sentences) > 0:
            tri_sentences.append(tri.sentences)
            has_relatives.extend(tri.relatives)

    # def filter_sentences_for_subgraph(one_subgraph: List[Triple], sids):
    #     filtered_sids = []
    #     doc_to_sids = dict()
    #     for sid in sids:
    #         tmp_docid, _ = sid.split(c_scorer.SENT_LINE2)[0], int(sid.split(c_scorer.SENT_LINE2)[1])
    #         if tmp_docid in doc_to_sids:
    #             doc_to_sids[tmp_docid].append(sid)
    #         else:
    #             doc_to_sids.update({tmp_docid: [sid]})
    #     for tri in one_subgraph:
    #         subj_short = uri_short_extract2(tri.subject)
    #         if subj_short in doc_to_sids:
    #             filtered_sids.extend(doc_to_sids[subj_short])
    #         if tri.object != "" and isURI(tri.object):
    #             obj_short = uri_short_extract2(tri.object)
    #             if obj_short in doc_to_sids:
    #                 filtered_sids.extend(doc_to_sids[obj_short])
    #     return filtered_sids


    # handle each subgraph separately
    # claim_subgraphs, _ = generate_triple_subgraphs(subgraph, claim_dict)
    sid_to_extend_sids = dict()
    # for claim_graph in claim_subgraphs:
    # isolated_entities = find_isolated_entity(claim_graph, claim_dict)
    # isolated_entity_phs = [ent['text'] for ent in isolated_entities]
    # candidate_sentences_sids = filter_sentences_for_subgraph(claim_dict)
    claim_linked_phrases_l = claim_dict["linked_phrase_l"]
    claim_linked_phs = [i['text'] for i in claim_linked_phrases_l]
    claim_linked_resources = []
    for t in subgraph:
        claim_linked_resources.append(t.subject)
        if t.object != "" and isURI(t.object):
            claim_linked_resources.append(t.object)
    claim_linked_resources = list(set(claim_linked_resources))
    for candidate_sent_sid in candidate_sentences:
        # extend_triples is in json format
        candidate_sent_text = ""
        doc_title, _ = candidate_sent_sid.split(c_scorer.SENT_LINE2)[0], int(candidate_sent_sid.split(c_scorer.SENT_LINE2)[1])
        candidate_context_graph, extend_triples_json = construct_subgraph_for_candidate2(candidate_sent_text,
                                                                                    doc_title=doc_title,
                                                                                    additional_phrase=claim_linked_phs,
                                                                                    additional_resources=claim_linked_resources,
                                                                                    bc=bc)
        extend_triples = []
        for idx_tri, tri in enumerate(extend_triples_json):
            tri['tri_id'] = f"{idx_tri}_extend"
            extend_triples.append(Triple(tri))
        extend_triples_dict = search_sentences_for_triples(extend_triples)  # tri.tri_id:list(set(tri.sentences))
        for ex_tri in extend_triples_dict:
            if len(extend_triples_dict[ex_tri]) > 0:
                sid_to_extend_sids.update({candidate_sent_sid: extend_triples_dict[ex_tri]})
    return sid_to_extend_sids, candidate_context_graph, extend_triples_json

    # for c_s in candidate_sentences:
    #     sent_context_graph, extend_triples = construct_subgraph_for_candidate(claim_dict, c_s['lines'], c_s['doc_id'])
    #     if len(extend_triples) < 1:
    #         continue
    #     extended_docs = search_entity_docs_for_triples(extend_triples)
    #     extended_sentences = search_triples_in_docs(extend_triples, extended_docs)
    #     for e_s in extended_sentences:
    #         extend_phrases = e_s['phrases']
    #         ts_phrases = c_s['phrases']
    #         bridged_phrases = list(set(extend_phrases + ts_phrases))
    #         update_relative_hash(relative_hash, bridged_phrases)
    #         c_s.extend_sentences.append(e_s.sid)
    #
    # # sort out minimal evidence set, then BERT filter h_links
    # possible_evidence_set = merge_sentences_and_generate_evidence_set(subgraph, candidate_sentences)
    # # BERT filter: (minimal set + h_link_sentence) VS claim
    # no_relatives_found = []
    # for i in relative_hash:
    #     if len(relative_hash[i]) == 0:
    #         no_relatives_found.append(i)
    # if len(no_relatives_found) > 0:
    #     for e_s in possible_evidence_set:
    #         h_link_docs = get_hlink_docs(e_s)
    #         extend_hlink_e = generate_expand_evidence_from_hlinks(e_s, h_link_docs)
    #         possible_evidence_set.extend(extend_hlink_e)
    # possible_evidence_set = list(set(possible_evidence_set))
    # # bert NLI model
    # pass


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


if __name__ == '__main__':
    folder = config.RESULT_PATH / "hardset2021"
    hardset_original = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    # candidate_docs = read_json_rows(folder / "candidate_docs.jsonl")
    # prepare_candidate_sents2_bert_dev(hardset_original, candidate_docs, folder)


    graph_data = read_json_rows(folder / "claim_graph.jsonl")
    # entity_data = read_json_rows(folder / "entity_doc.jsonl")
    # candidate_docs = read_json_rows(folder / "candidate_docs.jsonl")
    # prepare_candidate_sents3_from_triples(graph_data, entity_data, folder / "tri_ss.jsonl", folder / "tri_ss.log")

    tri_ss_data = read_json_rows(folder / "tri_ss.jsonl")
    # eval_tri_ss(hardset_original, tri_ss_data)
    bert_ss_data = read_json_rows(folder / "bert_ss_0.4_10.jsonl")
    # hit_eval(bert_ss_data, 10)
    # c_scorer.get_macro_ss_recall_precision(bert_ss_data, 5)
    # context_graph_data = read_json_rows(folder / "claim_graph.jsonl")
    generate_extend_sentences(graph_data, tri_ss_data, bert_ss_data, "")
    # prepare_evidence_set_for_bert_nli(hardset_original, bert_ss_data, tri_ss_data, context_graph_data, folder / "nli_sids.jsonl")


