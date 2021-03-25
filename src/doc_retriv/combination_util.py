from doc_retriv.SentenceEvidence import *
from utils.check_sentences import Evidences, sids_to_tuples
import copy
from typing import List
import itertools
from utils import c_scorer


def generate_sentence_combination(list_of_sentences: List):
    new_evidence_set = set()
    if len(list_of_sentences) > 8:
        max_s = 2
    elif len(list_of_sentences) < 3:
        max_s = len(list_of_sentences)
    else:
        max_s = 3
    for i in reversed(range(1, max_s + 1)):
        combination_set = itertools.combinations(list_of_sentences, i)
        for c in combination_set:
            sids = [s for s in c]
            raw_doc_ln = sids_to_tuples(sids)
            evid = Evidences(raw_doc_ln)
            new_evidence_set.add(evid)
    return list(new_evidence_set)


# each sentence may have different subgraphs, as one entity may linked to different resources
def generate_triple_subgraphs(list_of_triples: List[Triple]):
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
    # resource_to_phrase_dict = dict()
    # linked_phrases = claim_dict['linked_phrases_l']
    # for lp in linked_phrases:
    #     text = lp['text']
    #     links = lp['links']
    #     for l in links:
    #         res = l['URI']
    #         if res not in resource_to_phrase_dict:
    #             resource_to_phrase_dict.update({res: text})

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


def generate_triple_evidence_set(list_of_triples: List[Triple]):
    triple_set = []
    for tri in list_of_triples:
        exist = False
        for ts in triple_set:
            if any([tt.relatives == tri.relatives for tt in ts]):
                ts.append(tri)
                exist = True
        if not exist:
            triple_set.append([tri])

    def generate_evi_set(tris_sorted_by_rels_l, list_of_evidence: List[Evidences]):
        if len(tris_sorted_by_rels_l) == 0:
            return list_of_evidence
        else:
            tmp_triples_l = copy.deepcopy(tris_sorted_by_rels_l)
            triple_l = tmp_triples_l.pop(0)
            # new_evidence_l = list_of_evidence
            new_evidence_l = []
            if len(list_of_evidence) == 0:
                for triple in triple_l:
                    raw_doc_ln = sids_to_tuples(triple.sentences)
                    tmp_evidence_l = [Evidences([doc_ln]) for doc_ln in raw_doc_ln]
                    new_evidence_l.extend(tmp_evidence_l)
            else:
                # for tri_sid in triple.sentences:
                #     tmp_evidence_l = copy.deepcopy(list_of_evidence)
                #     for e in tmp_evidence_l:
                #         if len(e) < 3:
                #             e.add_sent_sid(tri_sid)
                #     new_evidence_l.extend(tmp_evidence_l)
                # new_evidence_l = list(set(new_evidence_l))
                for triple in triple_l:
                    for tri_sid in triple.sentences:
                        tmp_evidence_l = copy.deepcopy(list_of_evidence)
                        for e in tmp_evidence_l:
                            if len(e) < 3:
                                e.add_sent_sid(tri_sid)
                                new_evidence_l.append(e)
                if len(new_evidence_l) > 0:
                    new_evidence_l = list(set(new_evidence_l))
                else:
                    new_evidence_l = list_of_evidence
            return generate_evi_set(tmp_triples_l, new_evidence_l)
    return generate_evi_set(triple_set, [])


def merge_sentences_and_generate_evidence_set(linked_triples_with_sentences: List[Triple],
                                              candidate_sentences: List[str],
                                              sid2sids):
    subgraphs, subgraph_sentences_l = generate_triple_subgraphs(linked_triples_with_sentences)
    evidence_set_from_triple = []
    for sub_g in subgraphs:
        tmp_evi_from_tri_subgraph = generate_triple_evidence_set(sub_g)
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
    evidence_set_from_triple = generate_triple_evidence_set(subgraph)
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


