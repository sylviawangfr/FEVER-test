import unittest
from datetime import datetime
from dbpedia_sampler import bert_similarity
from graph_modules.dbpedia_ss_gat_sampler import DBpediaGATSampler
from utils.file_loader import *


class TestDB(unittest.TestCase):

    def test_bert(self):
        ps = ['Ashley Judd', 'Where the Heart Is', 'Stockard Channing', 'Joan Cusack',
                               'Natalie Portman', '2000', 'Sally Field', 'Keith David', 'Dylan Bruno', 'James Frain']
        start = datetime.now()
        for i in tqdm(range(10)):
            bert_similarity.get_phrase_embedding(ps)
        print(f"embedding time: {(datetime.now() - start).seconds}")

    def test_gat_sampler(self):
        start = datetime.now()
        data = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")[0:10]
        sample = DBpediaGATSampler(data)
        print(len(sample.graph_instances))
        print(f"sampling time: {(datetime.now() - start).seconds}")


if __name__ == '__main__':
    unittest.main()
