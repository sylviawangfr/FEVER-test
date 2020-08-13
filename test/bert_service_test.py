import unittest
from dbpedia_sampler import bert_similarity
from graph_modules.dbpedia_ss_gat_sampler import DBpediaGATSampler
from utils.file_loader import *
from datetime import datetime as dt


class TestDB(unittest.TestCase):

    def test_bert(self):
        ps = ['Ashley Judd', 'Where the Heart Is', 'Stockard Channing', 'Joan Cusack',
                               'Natalie Portman', '2000', 'Sally Field', 'Keith David', 'Dylan Bruno', 'James Frain']
        start = dt.now()
        for i in tqdm(range(10)):
            bert_similarity.get_phrase_embedding(ps)
        print(f"embedding time: {(dt.now() - start).seconds}")

    def test_gat_sampler(self):
        start = dt.now()
        data = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")[0:10]
        sample = DBpediaGATSampler(data, parallel=True)
        print(f"parallel graph pair count: {len(sample.graph_instances)}")
        print(f"parallel sampling time: {(dt.now() - start).seconds}")

        sample = DBpediaGATSampler(data, parallel=False)
        print(f"parallel graph pair count: {len(sample.graph_instances)}")
        print(f"parallel sampling time: {(dt.now() - start).seconds}")


if __name__ == '__main__':
    unittest.main()
