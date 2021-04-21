from utils.file_loader import read_json_rows, get_current_time_str, read_all_files, save_and_append_results
from doc_retriv.doc_retrieve_extend import *
from doc_retriv.ss import *

def dev_hardset_pipeline1(folder):
    original_data1 = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    prepare_candidate_doc1(original_data1, folder / "es_doc_10.jsonl", folder / "es_doc_10.log")
    del original_data1
    #
    data_with_es = read_json_rows(folder / "es_doc_10.jsonl")
    prepare_es_entity_links(data_with_es, folder / "es_entity_docs.jsonl")

    data_with_es_entities = read_json_rows(folder / "es_entity_docs.jsonl")
    original_data2 = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    assert (len(original_data2) == len(data_with_es))
    assert (len(data_with_es_entities) == len(original_data2))
    prepare_claim_graph(original_data2,
                        folder / "claim_graph.jsonl",
                        folder / "claim_graph.log",
                        data_with_entity_docs=data_with_es_entities,
                        data_with_es=data_with_es)
    del original_data2
    #
    original_data3 = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    data_context = read_json_rows(folder / "claim_graph.jsonl")
    prepare_candidate_doc2(original_data3, data_context, folder / "graph_resource_docs.jsonl",
                           folder / "graph_resource_docs.log")
    del original_data3
    #
    original_data4 = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    es_data = read_json_rows(folder / "es_doc_10.jsonl")
    ent_data = read_json_rows(folder / "graph_resource_docs.jsonl")
    assert (len(es_data) == len(original_data4) and (len(ent_data) == len(original_data4)))
    prepare_candidate_docs(original_data4, es_data, ent_data, folder / "candidate_docs.jsonl",
                           folder / "candidate_docs.log")

    candidate_docs = read_json_rows(folder / "candidate_docs.jsonl")
    prepare_candidate_sents2_bert_dev(hardset_original, candidate_docs, folder)

    graph_data = read_json_rows(folder / "claim_graph.jsonl")
    resource2docs_data = read_json_rows(folder / "graph_resource_docs.jsonl")
    prepare_candidate_sents3_from_triples(graph_data, resource2docs_data, folder / "tri_ss.jsonl", folder / "tri_ss.log")

    tri_ss_data = read_json_rows(folder / "tri_ss.jsonl")
    bert_ss_data = read_json_rows(folder / "bert_ss_0.4_10.jsonl")

    prepare_evidence_set_for_bert_nli(hardset_original, bert_ss_data, tri_ss_data, graph_data,
                                      folder / "nli_sids.jsonl")

def dev_hardset_pipeline2(folder):
    original_data1 = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    prepare_candidate_doc1(original_data1, folder / "es_doc_10.jsonl", folder / "es_doc_10.log")
    del original_data1
    #
    data_with_es = read_json_rows(folder / "es_doc_10.jsonl")

    data_bert_ss1 = prepare_candidate_sents2_bert_dev(hardset_original, data_with_es, folder)
    # prepare_ss_entity_links(data_with_es, folder / "es_entity_docs.jsonl")

    data_with_es_entities = read_json_rows(folder / "es_entity_docs.jsonl")
    original_data2 = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    assert (len(original_data2) == len(data_with_es))
    assert (len(data_with_es_entities) == len(original_data2))
    prepare_claim_graph(original_data2,
                        folder / "claim_graph.jsonl",
                        folder / "claim_graph.log",
                        data_with_entity_docs=data_with_es_entities,
                        data_with_es=data_with_es)
    del original_data2
    #
    original_data3 = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    data_context = read_json_rows(folder / "claim_graph.jsonl")
    prepare_candidate_doc2(original_data3, data_context, folder / "graph_resource_docs.jsonl",
                           folder / "graph_resource_docs.log")
    del original_data3
    #
    original_data4 = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    es_data = read_json_rows(folder / "es_doc_10.jsonl")
    ent_data = read_json_rows(folder / "graph_resource_docs.jsonl")
    assert (len(es_data) == len(original_data4) and (len(ent_data) == len(original_data4)))
    prepare_candidate_docs(original_data4, es_data, ent_data, folder / "candidate_docs.jsonl",
                           folder / "candidate_docs.log")

    candidate_docs = read_json_rows(folder / "candidate_docs.jsonl")
    prepare_candidate_sents2_bert_dev(hardset_original, candidate_docs, folder)

    graph_data = read_json_rows(folder / "claim_graph.jsonl")
    resource2docs_data = read_json_rows(folder / "graph_resource_docs.jsonl")
    prepare_candidate_sents3_from_triples(graph_data, resource2docs_data, folder / "tri_ss.jsonl",
                                          folder / "tri_ss.log")

    tri_ss_data = read_json_rows(folder / "tri_ss.jsonl")
    bert_ss_data = read_json_rows(folder / "bert_ss_0.4_10.jsonl")

    prepare_evidence_set_for_bert_nli(hardset_original, bert_ss_data, tri_ss_data, graph_data,
                                      folder / "nli_sids.jsonl")
