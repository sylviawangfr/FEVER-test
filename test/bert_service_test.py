import unittest
from datetime import datetime
from dbpedia_sampler import bert_similarity

class TestDB(unittest.TestCase):

    def test_bert(self):
        ps = ['Ashley Judd', 'Where the Heart Is', 'Stockard Channing', 'Joan Cusack',
                               'Natalie Portman', '2000', 'Sally Field', 'Keith David', 'Dylan Bruno', 'James Frain']
        start = datetime.now()
        for i in range(100):
            bert_similarity.get_phrase_embedding(ps)
        print(f"embedding time: {(datetime.now() - start).seconds}")


if __name__ == '__main__':
    unittest.main()
