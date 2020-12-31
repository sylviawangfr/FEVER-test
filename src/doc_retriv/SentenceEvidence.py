class SentenceEvidence:
    def __init__(self, sentence_dict):
        self.doc_id = sentence_dict['doc_id']
        self.sid = sentence_dict['sid']
        self.triple = sentence_dict['tri_id']
        self.score = sentence_dict['score']
        self.phrases = sentence_dict['phrases']
        self.text = sentence_dict['text']
        self.extend_sentences = sentence_dict['extend_sentences'] if 'extend_sentences' in sentence_dict else []
        self.h_links = sentence_dict['h_links']

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SentenceEvidence):
            return False
        return o.sid == self.sid


class Triple:
    def __init__(self, tri_dict):
        self.tri_id = tri_dict['tri_id']
        self.sentences = tri_dict['sentences'] if 'sentences' in tri_dict else []  # list of sid
        self.subject = tri_dict['subject']
        self.relation = tri_dict['relation']
        self.datatype = tri_dict['datatype'] if 'datatype' in tri_dict else ''
        self.object = tri_dict['object']
        self.keywords = tri_dict['keywords'] if 'keywords' in tri_dict else ''
        self.relatives = tri_dict['relatives'] if 'relatives' in tri_dict else ''
        self.text = tri_dict['text']
        self.exact_match = tri_dict['exact_match'] if 'exact_match' in tri_dict else False
        # self.score = tri_dict['score']
        self.URI = tri_dict['URI'] if 'URI' in tri_dict else tri_dict['subject']

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Triple):
            return False
        return o.tri_id == self.tri_id
