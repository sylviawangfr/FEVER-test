import json
import unicodedata
import config


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def read_json_rows(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(normalize(line.strip()))
            d_list.append(item)
    return d_list


def read_json(file):
    with open(file, encoding='utf-8', mode='r') as in_f:
        item = json.load(in_f)
        return item


def iter_baskets_contiguous(items, bunch_size):
    item_count = len(items)
    bunch_number_floor = len(items) // bunch_size
    bunch_number_celling = bunch_number_floor + 1
    for i in range(bunch_number_celling):
        start = i * bunch_size
        stop = (i + 1) * bunch_size
        stop = item_count if stop > item_count else stop
        yield [items[j] for j in range(start, stop)]


def test_iter_basket():
    p = iter_baskets_contiguous(range(50), 9)
    next(p)
    for v in p:
        print(v)
        assert(len(v) <= 9)

def test_read_json_rows():
    assert(read_json_rows(config.WIKI_PAGE_PATH / "wiki-001.jsonl"))





