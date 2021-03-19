from ES.es_search import  search_doc_id_and_keywords_in_sentences
from dbpedia_sampler.dbpedia_subgraph import construct_subgraph_for_candidate2
from dbpedia_sampler.dbpedia_triple_linker import filter_phrase_vs_two_hop, lookup_doc_id, filter_text_vs_one_hop, filter_triples, remove_duplicate_triples, add_outbound_single
from dbpedia_sampler import dbpedia_lookup, dbpedia_virtuoso
from BERT_test.ss_eval import *
from doc_retriv.doc_retrieve_extend import search_entity_docs_for_triples, is_media
from dbpedia_sampler.uri_util import uri_short_extract2, isURI, uri_short_extract
from utils.resource_manager import *
from utils.tokenizer_simple import get_lemma, is_capitalized
from tqdm import tqdm
from doc_retriv.combination_util import *


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
    sid2allhlinks = dict()
    all_doc_ln = [(i.split(c_scorer.SENT_LINE2)[0], int(i.split(c_scorer.SENT_LINE2)[1])) for i in all_sids]
    all_docs = list(set([i[0] for i in all_doc_ln]))
    for doc_id, ln in all_doc_ln:
        _, hlinks = get_text_and_hlinks(doc_id, ln)
        sid2allhlinks.update({f"{doc_id}{c_scorer.SENT_LINE2}{ln}": hlinks})
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

    return docid2sids, sid2hlinksids, sid2allhlinks


def add_linked_doc_ss(evi_set, sid2sids):
    add_linked_sid = []
    for evi in evi_set:
        if len(evi) < 3:
            e_sids = evi.to_sids()
            for ss in e_sids:
                if ss in sid2sids and len(sid2sids[ss]) > 0:
                    linked_sids = sid2sids[ss]
                    for l_s in linked_sids:
                        tmp_e_s = copy.deepcopy(evi)
                        tmp_e_s.add_sent_sid(l_s)
                        add_linked_sid.append(tmp_e_s)
    add_linked_sid = list(set(add_linked_sid))
    return add_linked_sid


def prepare_evidence_set_for_bert_nli(data_origin, data_with_bert_s,
                                      data_with_tri_s, data_with_context_graph, output_file):
    def get_bert_sids(scored_sentids, threshold=0.5):
        sids = []
        scores = dict()
        for i in scored_sentids:
            raw_sid = i[0]
            # score = i[-1]
            sid = raw_sid.replace(c_scorer.SENT_LINE, c_scorer.SENT_LINE2)
            sids.append(sid)
            scores.update({sid: i[2]})
        return sids, scores

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
                    if len(list(filter(lambda x: i in x, tri.keywords))) > 0:
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

    def has_media_linked(subg):
        for st in subg:
            subj = uri_short_extract(st.subject).lower()
            obj = uri_short_extract(st.object).lower()
            tri_text = st.text.lower()
            for i in media_entity:
                if i.lower() in [subj, obj, tri_text]:
                    return True
        return False

    with tqdm(total=len(data_origin), desc=f"generating nli candidate") as pbar:
        for idx, example in enumerate(data_origin):
            #   379, 402, 646, 910, 976, 993, 1043, 1058, 1219, 1446, 1554, 1591, 1616, 1723
            # if idx < 1723:
            #     continue
            # ["Soul_Food_-LRB-film-RRB-<SENT_LINE>0", 1.4724552631378174, 0.9771634340286255]
            bert_s, bert_sid2score = get_bert_sids(data_with_bert_s[idx]['scored_sentids'])
            triples = [Triple(t_dict) for t_dict in data_with_tri_s[idx]['triples']]
            claim_dict = data_with_context_graph[idx]['claim_dict']
            linked_phrases = claim_dict['linked_phrases_l']
            linked_entities = [ent['text'] for ent in linked_phrases]
            media_entity = [ent for ent in linked_entities if is_media(ent)]
            candidate_sid_sets = []
            tri_s = list(set([s for tt in triples for s in tt.sentences]))
            bert_and_tri_s = list(set(bert_s) | set(tri_s))
            all_docid2sids, all_sid2linkedsids, sid2allhlinks = get_docid_to_sids_hlinks(bert_and_tri_s)
            if len(bert_and_tri_s) > 20:
                bert_docid2sids, bert_sid2linkedsids = get_docid_to_sids_hlinks(bert_s)
                docid_dict_to_calc = bert_docid2sids
                sid2linkedsids_to_calc = bert_sid2linkedsids
            else:
                docid_dict_to_calc = all_docid2sids
                sid2linkedsids_to_calc = all_sid2linkedsids

            # for linked_ent in linked_entities:
            #     for docid in docid_dict_to_calc:
            #         if linked_ent.lower() in convert_brc(docid).replace("_", " ").lower():
            #             tmp_sids = docid_dict_to_calc[docid]
            #             sid_combination = generate_sentence_combination(tmp_sids)
            #             candidate_sid_sets.extend(sid_combination)
            for docid in docid_dict_to_calc:
                tmp_sids = docid_dict_to_calc[docid]
                sid_combination = generate_sentence_combination(tmp_sids)
                candidate_sid_sets.extend(sid_combination)

            candidate_sid_sets.extend(add_linked_doc_ss(candidate_sid_sets, sid2linkedsids_to_calc))
            candidate_sid_sets = list(set(candidate_sid_sets))
            linked_level = 0    # [0: not linked, 1: partial linked, 2: well linked]
            if len(triples) > 0:
                subgraphs, _ = generate_triple_subgraphs(triples)
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
                        # tmp_sid_sets = generate_triple_evidence_set(good_subgraph, [])
                        tmp_sid_sets = generate_triple_evidence_set(good_subgraph)
                        candidate_sid_sets.extend(tmp_sid_sets)
                    linked_level = 2
                elif len(partial_linked_idx) > 0:
                    for k, item in enumerate(partial_linked_idx):
                        subgraph_sids = all_subgraph_sids[item]
                        if all([len(s) == 0 for s in subgraph_sids]):
                            continue
                        else:
                            subgraph = subgraphs[item]
                            # tmp_sid_sets = generate_triple_evidence_set(subgraph, [])
                            tmp_sid_sets = generate_triple_evidence_set(subgraph)
                            if len(tmp_sid_sets) == 0:
                                continue
                            not_linked_phrases = partial_linked_no_relatives_phrases[k]
                            if any(['film' in tri.subject for tri in subgraph]):
                                print("debug")
                            if len(not_linked_phrases) > 3 \
                                    or (len(media_entity) > 0 and not has_media_linked(subgraph))\
                                    or len(not_linked_phrases) / len(linked_phrases) > 0.5:
                                continue
                            extend_evi = add_linked_doc_ss(tmp_sid_sets, all_sid2linkedsids)
                            if len(extend_evi) > 0:
                                tmp_sid_sets.extend(extend_evi)
                            else:
                                # 1. candidate tri two hop
                                extend_evi = extend_evidence_two_hop_nodes(not_linked_phrases, subgraph, tmp_sid_sets)
                                tmp_sid_sets.extend(extend_evi)
                            candidate_sid_sets.extend(tmp_sid_sets)
                            linked_level = 1
            else:
                # count_capitalized_entities = []
                # for i in linked_entities:
                #     if i not in count_capitalized_entities\
                #             and len(list(filter(lambda x: i in x or x in i, count_capitalized_entities))) == 0:
                #         count_capitalized_entities.append(i)
                most_similar_sent = [s for s in bert_s if bert_sid2score[s] > 0.95
                                     and s.split(c_scorer.SENT_LINE2)[0].replace('_', ' ') in linked_entities][0:2]
                if len(most_similar_sent) > 0: # and len(count_capitalized_entities) > 1:
                    #   only check the most similar sentences
                    tmp_docid2sids = dict()
                    for d in all_docid2sids:
                        sids = all_docid2sids[d]
                        tmp_sids = [ts for ts in sids if ts in most_similar_sent]
                        if len(tmp_sids) > 0:
                            tmp_docid2sids.update({d: tmp_sids})
                    # 1. candidate sent two hop
                    extend_sid_set = extend_evidence_two_hop_sentences(claim_dict, tmp_docid2sids, all_sid2linkedsids)
                    if len(extend_sid_set) > 0:
                        candidate_sid_sets.extend(extend_sid_set)
                        linked_level = 1
                    else:
                        #   2. hlink sids
                        extend_sid_set = []
                        for sid in most_similar_sent:
                            extend_sid_set.extend(extend_hlink_sids(sid, sid2allhlinks[sid]))
                        candidate_sid_sets.extend(extend_sid_set)
                        linked_level = 0
                else:
                    linked_level = 0
            candidate_sid_sets = list(set(candidate_sid_sets))
            example.update({'nli_sids': [e.to_sids() for e in candidate_sid_sets], 'linked_level': linked_level})
            pbar.update(1)
    save_intermidiate_results(data_origin, output_file)
    print("Done with nli_sids")


def extend_hlink_sids(sid, hlinks, max_number=5):
    extend_sid_set = []
    db = FeverDBResource()
    cursor = db.get_cursor()
    for h in hlinks:
        tmp_docid = h.replace(' ', '_')
        text_l, hsid_l = get_all_sent_by_doc_id(cursor, tmp_docid)
        if len(hsid_l) == 0:
            h_res_l = dbpedia_lookup.lookup_label_exact_match(h)
            for res in h_res_l:
                uri = res['URI']
                redirect_entities = dbpedia_virtuoso.get_resource_redirect(uri)
                for redir_ent in redirect_entities:
                    text_l, hsid_l = get_all_sent_by_doc_id(cursor, redir_ent)
        hsid_l = hsid_l[0:max_number]
        for hsid in hsid_l:
            extend_sid_set.append([(sid, hsid)])
    extend_evi_set = [Evidences(ss) for ss in extend_sid_set]
    return extend_evi_set


def extend_evidence_two_hop_nodes(phrases, subgraph, subgraph_evi_set):
    extend_triples_jsonl = filter_phrase_vs_two_hop(phrases, subgraph)
    if len(extend_triples_jsonl) == 0:
        return []
    extend_triples = []
    for idx_tri, tri in enumerate(extend_triples_jsonl):
        tri['tri_id'] = f"{idx_tri}_extend"
        extend_triples.append(Triple(tri))
    search_triples_in_docs(extend_triples)  # {tri_id: sids}
    tri2extri = {t.tri_id: [] for t in subgraph}
    for tri in subgraph:
        for ex_tri in extend_triples:
            if tri.object == ex_tri.subject:
                tri2extri[tri.tri_id].append(ex_tri)
    sid2extendsids = dict()
    triid2tri = {t.tri_id: t for t in subgraph}
    for tri_id in tri2extri:
        if len(tri2extri[tri_id]) > 0:
            tri_sids = triid2tri[tri_id].sentences
            ex_tris = tri2extri[tri_id]
            ex_sids = list(set([ss for tt in ex_tris for ss in tt.sentences]))
            for ts in tri_sids:
                if ts in sid2extendsids:
                    sid2extendsids[ts].extend(ex_sids)
                else:
                    sid2extendsids.update({ts:ex_sids})
    extend_evi_set = add_linked_doc_ss(subgraph_evi_set, sid2extendsids)
    return extend_evi_set


def extend_evidence_two_hop_sentences(claim_dict, docid2sids, sid2linkedsids):
    claim_linked_resources = claim_dict["linked_phrases_l"]
    claim_linked_phs = [i['text'] for i in claim_linked_resources]
    embedding_hash = dict()
    sid_to_extend_sids = dict()
    has_checked = []
    def record_checked(phs, hls):
        for i in itertools.product(phs, hls):
            if (i[0], i[1]) not in has_checked:
                has_checked.append((i[0], i[1]))

    def get_phs_and_hlinks_res_to_match(phs, hls):
        p_h_tomatch = []
        for i in phs:
            h_l = []
            for h in hls:
                if (i, h) not in has_checked:
                    h_l.append(h)
            p_h_tomatch.append((i, h_l))
        return p_h_tomatch

    hlink_resources = dict()
    sid2hlinks = dict()
    to_extend_candidate_sentences = []
    for docid in docid2sids:
        candidate_sentences = docid2sids[docid]
        all_extend_triples_jsonl = []
        for candidate_sent_sid in candidate_sentences:
            tmp_extend_triples_jsonl = []
            if candidate_sent_sid in sid2linkedsids or any([candidate_sent_sid in x for x in sid2linkedsids.values()]):
                continue
            # extend_triples is in json format
            doc_title, ln = candidate_sent_sid.split(c_scorer.SENT_LINE2)[0], int(candidate_sent_sid.split(c_scorer.SENT_LINE2)[1])
            doc_id_clean = convert_brc(doc_title).replace('_', ' ').lower()
            if all([(x.lower() not in doc_id_clean and doc_id_clean not in x.lower()) for x in claim_linked_phs]):
                continue
            candidate_sent_text, hlinks = get_text_and_hlinks(doc_title, ln)
            sid2hlinks.update({candidate_sent_sid: hlinks})
            if len(hlinks) == 0:
                continue
            for hl in hlinks:
                if hl not in hlink_resources:
                    extend_resources = lookup_doc_id(hl, [hl])
                    if len(extend_resources) == 0:
                        continue
                    add_outbound_single(extend_resources)
                    hlink_resources.update({hl: extend_resources})
            to_match_phrases = []
            for p in claim_linked_phs:
                csid_clean = convert_brc(candidate_sent_sid.split(c_scorer.SENT_LINE2)[0]).replace("_", " ").lower()
                p_lower = p.lower()
                if p_lower != csid_clean and all([x not in csid_clean for x in get_lemma(p_lower)]):
                    to_match_phrases.append(p)
            to_check_p_hks_combinations = get_phs_and_hlinks_res_to_match(to_match_phrases, hlinks)
            for phc in to_check_p_hks_combinations:
                single_ph = phc[0]
                ph_hlinks = phc[1]
                to_match_hlink_res = [hlink_resources[h] for h in ph_hlinks if h in hlink_resources]
                extend_triples_jsonl = filter_text_vs_one_hop([single_ph], to_match_hlink_res, embedding_hash)
                tmp_extend_triples_jsonl.extend(extend_triples_jsonl)
            if len(tmp_extend_triples_jsonl) == 0:
                continue
            else:
                filtered_triples = remove_duplicate_triples(tmp_extend_triples_jsonl)
                filtered_triples = filter_triples(filtered_triples)
                all_extend_triples_jsonl.extend(filtered_triples)
            record_checked(to_match_phrases, hlinks)
        if len(all_extend_triples_jsonl) == 0:
            continue
        extend_triples = []
        for idx_tri, tri in enumerate(all_extend_triples_jsonl):
            tri['tri_id'] = f"{idx_tri}_extend"
            extend_triples.append(Triple(tri))
        tri_sentence_dict = search_triples_in_docs(extend_triples)  # {tri_id: sids}
        all_e_sids = []
        for ex_sids in tri_sentence_dict.values():
            all_e_sids.extend(ex_sids)
        all_e_sids = list(set(all_e_sids))
        if len(all_e_sids) > 0:
            for candidate_sent_sid in candidate_sentences:
                css_hlinks = sid2hlinks[candidate_sent_sid]
                if len(css_hlinks) == 0:
                    continue
                css_extend_sids = []
                for h in css_hlinks:
                    css_extend_sids.extend([x for x in all_e_sids if h.lower() in convert_brc(x).replace('_', ' ').lower()])
                css_extend_sids = list(set(css_extend_sids))
                if len(css_extend_sids) > 0:
                    sid_to_extend_sids.update({candidate_sent_sid: css_extend_sids})
                    to_extend_candidate_sentences.append(candidate_sent_sid)
    if len(to_extend_candidate_sentences) == 0:
        return []
    candidate_evi_sets = [Evidences([(ss.split(c_scorer.SENT_LINE2)[0], ss.split(c_scorer.SENT_LINE2)[1])]) for ss in to_extend_candidate_sentences]
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


def extend_evidence_two_hop_sentences2(claim_dict, docid2sids, sid2linkedsids):
    claim_linked_resources = claim_dict["linked_phrases_l"]
    claim_linked_phs = [i['text'] for i in claim_linked_resources]
    embedding_hash = dict()
    sid_to_extend_sids = dict()
    has_checked = []
    def record_checked(phs, hls):
        for i in itertools.product(phs, hls):
            if (i[0], i[1]) not in has_checked:
                has_checked.append((i[0], i[1]))

    def get_phs_and_hlinks_res_to_match(phs, hls):
        p_h_tomatch = []
        for i in phs:
            h_l = []
            for h in hls:
                if (i, h) not in has_checked:
                    h_l.append(h)
            p_h_tomatch.append((i, h_l))
        return p_h_tomatch

    hlink_resources = dict()
    sid2hlinks = dict()
    to_extend_candidate_sentences = []
    for docid in docid2sids:
        candidate_sentences = docid2sids[docid]
        all_extend_triples_jsonl = []
        for candidate_sent_sid in candidate_sentences:
            tmp_extend_triples_jsonl = []
            if candidate_sent_sid in sid2linkedsids or any([candidate_sent_sid in x for x in sid2linkedsids.values()]):
                continue
            # extend_triples is in json format
            doc_title, ln = candidate_sent_sid.split(c_scorer.SENT_LINE2)[0], int(candidate_sent_sid.split(c_scorer.SENT_LINE2)[1])
            doc_id_clean = convert_brc(doc_title).replace('_', ' ').lower()
            if all([(x.lower() not in doc_id_clean and doc_id_clean not in x.lower()) for x in claim_linked_phs]):
                continue
            candidate_sent_text, hlinks = get_text_and_hlinks(doc_title, ln)
            sid2hlinks.update({candidate_sent_sid: hlinks})
            if len(hlinks) == 0:
                continue
            for hl in hlinks:
                if hl not in hlink_resources:
                    extend_resources = lookup_doc_id(hl, [hl])
                    if len(extend_resources) == 0:
                        continue
                    add_outbound_single(extend_resources)
                    hlink_resources.update({hl: extend_resources})
            to_match_phrases = []
            for p in claim_linked_phs:
                csid_clean = convert_brc(candidate_sent_sid.split(c_scorer.SENT_LINE2)[0]).replace("_", " ").lower()
                p_lower = p.lower()
                if p_lower != csid_clean and all([x not in csid_clean for x in get_lemma(p_lower)]):
                    to_match_phrases.append(p)
            to_check_p_hks_combinations = get_phs_and_hlinks_res_to_match(to_match_phrases, hlinks)
            for phc in to_check_p_hks_combinations:
                single_ph = phc[0]
                ph_hlinks = phc[1]
                to_match_hlink_res = [hlink_resources[h] for h in ph_hlinks if h in hlink_resources]
                extend_triples_jsonl = filter_text_vs_one_hop([single_ph], to_match_hlink_res, embedding_hash, threshold=0.6)
                tmp_extend_triples_jsonl.extend(extend_triples_jsonl)
            if len(tmp_extend_triples_jsonl) == 0:
                continue
            else:
                filtered_triples = remove_duplicate_triples(tmp_extend_triples_jsonl)
                filtered_triples = filter_triples(filtered_triples)[0:3]
                all_extend_triples_jsonl.extend(filtered_triples)
            record_checked(to_match_phrases, hlinks)
        if len(all_extend_triples_jsonl) == 0:
            continue
        extend_triples = []
        for idx_tri, tri in enumerate(all_extend_triples_jsonl):
            tri['tri_id'] = f"{idx_tri}_extend"
            extend_triples.append(Triple(tri))
        tri_sentence_dict = search_triples_in_docs(extend_triples)  # {tri_id: sids}
        all_e_sids = []
        for ex_sids in tri_sentence_dict.values():
            all_e_sids.extend(ex_sids)
        if len(all_e_sids) > 0:
            for candidate_sent_sid in candidate_sentences:
                css_hlinks = sid2hlinks[candidate_sent_sid]
                if len(css_hlinks) == 0:
                    continue
                css_extend_sids = []
                for h in css_hlinks:
                    css_extend_sids.extend([x for x in all_e_sids if h.lower() in convert_brc(x).replace('_', ' ').lower()])
                css_extend_sids = list(set(css_extend_sids))
                if len(css_extend_sids) > 0:
                    sid_to_extend_sids.update({candidate_sent_sid: list(set(css_extend_sids))})
                    to_extend_candidate_sentences.append(candidate_sent_sid)
    if len(to_extend_candidate_sentences) == 0:
        return []
    candidate_evi_sets = generate_sentence_combination(to_extend_candidate_sentences)
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
    candidate_docs = read_json_rows(folder / "candidate_docs.jsonl")
    prepare_candidate_sents2_bert_dev(hardset_original, candidate_docs, folder)

    graph_data = read_json_rows(folder / "claim_graph.jsonl")
    resource2docs_data = read_json_rows(folder / "graph_resource_docs.jsonl")
    prepare_candidate_sents3_from_triples(graph_data, resource2docs_data, folder / "tri_ss.jsonl", folder / "tri_ss.log")

    tri_ss_data = read_json_rows(folder / "tri_ss.jsonl")
    bert_ss_data = read_json_rows(folder / "bert_ss_0.4_10.jsonl")

    # generate_candidate_graphs(graph_data, tri_ss_data, bert_ss_data,
    #                           folder / "sids.jsonl", folder / "sid2graph.jsonl",
    #                           folder / "sids.log", folder / "sid2graph.log")
    #
    # sid2sids_data = read_json_rows(folder / "sids.jsonl")
    # docs_data = read_json_rows(folder/ "es_doc_10.jsonl")
    prepare_evidence_set_for_bert_nli(hardset_original, bert_ss_data, tri_ss_data, graph_data, folder / "nli_sids.jsonl")


