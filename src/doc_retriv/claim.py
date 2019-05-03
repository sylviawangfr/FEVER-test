class ClaimDocRetr(object):
    # ClaimDocRetr is a record of
    def __init__(self, evidences):
        evidences_set = set()
        for doc_id, line_num in evidences:
            if doc_id is not None and line_num is not None:
                evidences_set.add((doc_id, line_num))

        evidences_list = sorted(evidences_set, key=lambda x: (x[0], x[1]))
        # print(evidences_list)
        self.evidences_list = evidences_list