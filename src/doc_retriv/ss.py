from ES.es_search import  search_doc_id_and_keywords_in_sentences
# from utils.c_scorer import *
from utils.fever_db import get_evidence
from dbpedia_sampler.dbpedia_subgraph import construct_subgraph_for_candidate2
from dbpedia_sampler.dbpedia_triple_linker import filter_phrase_vs_two_hop, lookup_doc_id, filter_text_vs_one_hop, filter_triples, add_outbounds
from doc_retriv.SentenceEvidence import *
from utils.check_sentences import Evidences, sids_to_tuples
import copy
from typing import List
import itertools
from BERT_test.ss_eval import *
from doc_retriv.doc_retrieve_extend import search_entity_docs_for_triples
from dbpedia_sampler.uri_util import uri_short_extract2, isURI
from utils.resource_manager import *
import itertools
from tqdm import tqdm
from collections import Counter


def filter_bert_claim_vs_sents(claim, docs):
    pass


def prepare_candidate_sents2_bert_dev(original_data, data_with_candidate_docs, output_folder):
    paras = bert_para.PipelineParas()
    paras.data_from_pred = True
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
                                      data_with_tri_s, data_with_context_graph, output_file):
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

    def get_not_linked_phrases(sub_graph, relative_hash=None):
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
        return no_relatives_found

    def add_linked_doc_ss(evi_set):
        add_linked_sid = []
        for evi in evi_set:
            if len(evi) < 3:
                e_sids = evi.to_sids()
                for ss in e_sids:
                    if ss in sid2linkedsids and len(sid2linkedsids[ss]) > 0:
                        linked_sids = sid2linkedsids[ss]
                        for l_s in linked_sids:
                            tmp_e_s = copy.deepcopy(evi)
                            tmp_e_s.add_sent_sid(l_s)
                            add_linked_sid.append(tmp_e_s)
        return add_linked_sid

    with tqdm(total=len(data_origin), desc=f"generating nli candidate") as pbar:
        for idx, example in enumerate(data_origin):
            # if idx < 62:
            #     continue
            # ["Soul_Food_-LRB-film-RRB-<SENT_LINE>0", 1.4724552631378174, 0.9771634340286255]
            bert_s = get_bert_sids(data_with_bert_s[idx]['scored_sentids'])
            triples = [Triple(t_dict) for t_dict in data_with_tri_s[idx]['triples']]
            claim_dict = data_with_context_graph[idx]['claim_dict']
            linked_phrases = claim_dict['linked_phrases_l']
            linked_entities = [ent['text'] for ent in linked_phrases]
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
                partial_linked_no_relatives_phrases = []
                all_subgraph_sids = []
                for subgraph in subgraphs:
                    tmp_sids = []
                    for tt in subgraph:
                        tmp_sids.append(tt.sentences)
                    all_subgraph_sids.append(tmp_sids)
                for i, subgraph in enumerate(subgraphs):
                    subgraph_sids = all_subgraph_sids[i]
                    no_relative = get_not_linked_phrases(subgraph, claim_relative_hash)
                    if len(no_relative) == 0 and all([len(s) > 0 for s in subgraph_sids]):
                        well_linked_sg_idx.append(i)
                    else:
                        partial_linked_idx.append(i)
                        partial_linked_no_relatives_phrases.append(no_relative)
                if len(well_linked_sg_idx) > 0:
                    for j in well_linked_sg_idx:
                        good_subgraph = subgraphs[j]
                        tmp_sid_sets = generate_triple_sentence_combination(good_subgraph, [])
                        candidate_sid_sets.extend(tmp_sid_sets)
                elif len(partial_linked_idx) > 0:
                    for k, item in enumerate(partial_linked_idx):
                        subgraph_sids = all_subgraph_sids[item]
                        if all([len(s) == 0 for s in subgraph_sids]):
                            continue
                        else:
                            subgraph = subgraphs[item]
                            tmp_sid_sets = generate_triple_sentence_combination(subgraph, [])
                            extend_sid_set = []
                            extend_evi = add_linked_doc_ss(tmp_sid_sets)
                            if len(extend_evi) > 0:
                                extend_sid_set.extend(extend_evi)
                                tmp_sid_sets.extend(extend_sid_set)
                            else:
                                # 1. candidate tri two hop
                                # get not linked phrase, match two hop nodes
                                not_linked_phrases = partial_linked_no_relatives_phrases[k]
                                extend_evi = extend_evidence_two_hop_nodes(not_linked_phrases, subgraph)
                                if len(extend_evi) > 0:
                                    tmp_sid_sets = extend_evi
                            candidate_sid_sets.extend(tmp_sid_sets)
            else:
                if len(bert_and_tri_s) > 0 and len(linked_phrases) > 1:
                    # 2. candidate sent two hop
                    extend_sid_set = extend_evidence_two_hop_sentences(claim_dict, bert_s)
                    candidate_sid_sets.extend(extend_sid_set)

            candidate_sid_sets = list(set(candidate_sid_sets))
            example.update({'nli_sids': [e.to_sids() for e in candidate_sid_sets]})
            pbar.update(1)
    save_intermidiate_results(data_origin, output_file)
    print("Done with nli_sids")


def extend_evidence_two_hop_nodes(phrases, subgraph):
    extend_triples_jsonl = filter_phrase_vs_two_hop(phrases, subgraph)
    if len(extend_triples_jsonl) == 0:
        return []
    extend_triples = []
    for idx_tri, tri in enumerate(extend_triples_jsonl):
        tri['tri_id'] = f"{idx_tri}_extend"
        extend_triples.append(Triple(tri))
    search_triples_in_docs(extend_triples)  # {tri_id: sids}
    tmp_extend_subgraph = copy.deepcopy(subgraph)
    tmp_extend_subgraph.extend(extend_triples)
    extend_evi_set = generate_triple_sentence_combination(tmp_extend_subgraph, [])
    return extend_evi_set


def extend_evidence_two_hop_sentences(claim_dict, candidate_sentences):
    sid_to_extend_sids = dict()
    sid2graph_l = dict()
    claim_linked_resources = claim_dict["linked_phrases_l"]
    claim_linked_phs = [i['text'] for i in claim_linked_resources]
    embedding_hash = dict()
    sid_to_extend_sids = dict()
    has_checked = []

    def record_checked(phs, hls):
        for i in itertools.product(phs, hls):
            if (i[0], i[1]) not in has_checked:
                has_checked.append((i[0], i[1]))

    def get_phs_and_hlinks_to_match(phs, hls):
        phs_tomatch = []
        hls_tomatch = []
        for i in itertools.product(phs, hls):
            if (i[0], i[1]) not in has_checked:
                phs_tomatch.append(i[0])
                hls_tomatch.append(i[1])
        return phs_tomatch, hls_tomatch

    for candidate_sent_sid in candidate_sentences:
        # extend_triples is in json format
        doc_title, ln = candidate_sent_sid.split(c_scorer.SENT_LINE2)[0], int(candidate_sent_sid.split(c_scorer.SENT_LINE2)[1])
        candidate_sent_text, hlinks = get_text_and_hlinks(doc_title, ln)
        if len(hlinks) == 0:
            continue
        to_match_phrases = [p for p in claim_linked_phs if
                            p.lower() != convert_brc(candidate_sent_sid.split(c_scorer.SENT_LINE2)[0]).replace("_", " ").lower()]
        to_match_phrases, hlinks = get_phs_and_hlinks_to_match(to_match_phrases, hlinks)
        all_extend_triples_jsonl = []
        hlink_resources = []

        for hl in hlinks:
            extend_resources = lookup_doc_id(hl, [hl])
            if len(extend_resources) == 0:
                continue
            hlink_resources.append(extend_resources)
        add_outbounds(hlink_resources)
        extend_triples_jsonl = filter_text_vs_one_hop(to_match_phrases, hlink_resources, embedding_hash, threshold=0.6)
        record_checked(to_match_phrases, hlinks)
        has_checked.append((to_match_phrases, hlinks))
        if len(extend_triples_jsonl) == 0:
            continue
        all_extend_triples_jsonl.extend(extend_triples_jsonl)
        filtered_triples = filter_triples(all_extend_triples_jsonl)
        extend_triples = []
        for idx_tri, tri in enumerate(filtered_triples):
            tri['tri_id'] = f"{idx_tri}_extend"
            extend_triples.append(Triple(tri))
        tri_sentence_dict = search_triples_in_docs(extend_triples)  # {tri_id: sids}
        all_e_sids = []
        for ex_sids in tri_sentence_dict.values():
            all_e_sids.extend(ex_sids)
        if len(all_e_sids) > 0:
            sid_to_extend_sids.update({candidate_sent_sid: list(set(all_e_sids))})
    candidate_evi_sets = generate_sentence_combination(candidate_sentences)
    extend_evidence_set = []
    for evi in candidate_evi_sets:
        if len(evi) > 3:
            continue
        evi_sids = evi.to_sids()
        for sid in evi_sids:
            if sid in sid_to_extend_sids:
                extend_sids = sid_to_extend_sids[sid]
                for es in extend_sids:
                    tmp_evi = copy.deepcopy(evi)
                    tmp_evi.add_sent_sid(es)
                    extend_evidence_set.append(tmp_evi)
    candidate_evi_sets.extend(extend_evidence_set)
    return candidate_evi_sets


# the resource_to_doc_dict is constructed in advance for performance concern
def search_triples_in_docs(triples: List[Triple], docs:dict=None):  #  list[Triple]
    if docs is None or len(docs) == 0:
        docs = search_entity_docs_for_triples(triples)
    for tri in triples:
        resource_docs = []
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
                             or (tri1_res_subj == tri2_res_obj
                                 and ('http://dbpedia.org/resource/' in tri2_res_obj
                                      or 'http://dbpedia.org/ontology/' in tri2_res_obj))):
                    return True
                if tri1_text != tri2_text \
                        and (tri1_res_obj == tri2_res_subj
                             or tri1_res_subj == tri2_res_obj
                             or (tri2_res_obj == tri1_res_obj
                                 and ('http://dbpedia.org/resource/' in tri2_res_obj
                                      or 'http://dbpedia.org/ontology/' in tri2_res_obj))):
                    return True
        return False

    def merge_tri_from_res1_to_res2(tri_l1_tomerge, tri_l2_desc, tri_dict):
        relative2 = list(set([r for t2 in tri_l2_desc for r in tri_dict[t2].relatives]))
        desc_keywords = list(set([k for x in tri_l2_desc for k in tri_dict[x].keywords]))
        desc_rel_and_keywords = list(set(relative2) | set(desc_keywords))
        for t1 in tri_l1_tomerge:
            t1_rel = tri_dict[t1].relatives
            if any([len(list(filter(lambda x: k1.lower() in x.lower() or x.lower() in k1.lower(), desc_rel_and_keywords))) == 0 for k1 in t1_rel]):
                tri_l2_desc.append(t1)
            relative2 = list(set([r for t2 in tri_l2_desc for r in tri_dict[t2].relatives]))
            desc_keywords = list(set([k for x in tri_l2_desc for k in tri_dict[x].keywords]))
            desc_rel_and_keywords = list(set(relative2) | set(desc_keywords))
        return tri_l2_desc

    def generate_subgraphs(tri_l, tri_dict, linked_l):
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
                        merge_tri_from_res1_to_res2(r_t, dup_linked_t2, tri_dict)
                        if dup_linked_t2 not in linked_l:
                            linked_l.append(dup_linked_t2)
                    if has_overlap_resource(linked_t, r_t, tri_dict):
                        dup_r_t = copy.deepcopy(r_t)
                        merge_tri_from_res1_to_res2(linked_t, dup_r_t, tri_dict)
                        if dup_r_t not in linked_l:
                            linked_l.append(dup_r_t)
                    if r_t not in linked_l:
                        linked_l.append(r_t)
            return generate_subgraphs(tri_l, tri_dict, linked_l)

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
        subj = tri.subject
        if subj in resource_to_tris_dict:
            resource_to_tris_dict[subj].append(tri.tri_id)
        else:
            resource_to_tris_dict.update({subj: [tri.tri_id]})

    all_resource_tri_sets = list(resource_to_tris_dict.values())
    subgraphs = generate_subgraphs(all_resource_tri_sets, tri_id_to_tri_dict, [])
    subgraphs = [list(set(s)) for s in subgraphs]
    remove_dup = []
    for sg in subgraphs:
        sg.sort()
        if sg not in remove_dup:
            remove_dup.append(sg)
    subgraphs = remove_dup
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
        # new_evidence_l = list_of_evidence
        new_evidence_l = []
        if len(list_of_evidence) == 0:
            raw_doc_ln = sids_to_tuples(triple.sentences)
            new_evidence_l = [Evidences([doc_ln]) for doc_ln in raw_doc_ln]
        else:
            # for tri_sid in triple.sentences:
            #     tmp_evidence_l = copy.deepcopy(list_of_evidence)
            #     for e in tmp_evidence_l:
            #         e.add_sent_sid(tri_sid)
            #     new_evidence_l.extend(tmp_evidence_l)
            #     new_evidence_l = list(set(new_evidence_l))
            for tri_sid in triple.sentences:
                tmp_evidence_l = copy.deepcopy(list_of_evidence)
                for e in tmp_evidence_l:
                    e.add_sent_sid(tri_sid)
                    new_evidence_l.append(e)
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
        extend_triples_dict = search_triples_in_docs(extend_triples)  # tri.tri_id:list(set(tri.sentences))
        all_e_sids = []
        for ex_sids in extend_triples_dict.values():
            if len(ex_sids) > 0:
                all_e_sids.extend(ex_sids)
        sid_to_extend_sids.update({candidate_sent_sid: list(set(all_e_sids))})
    return sid_to_extend_sids, sid2graph_l


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

    # generate_candidate_graphs(graph_data, tri_ss_data, bert_ss_data,
    #                           folder / "sids.jsonl", folder / "sid2graph.jsonl",
    #                           folder / "sids.log", folder / "sid2graph.log")
    #
    # sid2sids_data = read_json_rows(folder / "sids.jsonl")
    # docs_data = read_json_rows(folder/ "es_doc_10.jsonl")
    prepare_evidence_set_for_bert_nli(hardset_original, bert_ss_data, tri_ss_data, graph_data, folder / "nli_sids.jsonl")


