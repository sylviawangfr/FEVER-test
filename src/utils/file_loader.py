import json
import unicodedata


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def read_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)
    return d_list


def iter_baskets_contiguous(items, bunch_size):
    item_count = len(items)
    bunch_number_floor = len(items) // bunch_size
    bunch_number_celling = bunch_number_floor + 1
    print(bunch_number_celling)
    for i in range(bunch_number_celling):
        print(range(i * bunch_size, (i + 1) * bunch_size))
        start = i * bunch_size
        stop = (i + 1) * bunch_size
        stop = item_count if stop > item_count else stop
        yield [items[j] for j in range(start, stop)]


def test():
    p = iter_baskets_contiguous(range(50), 9)
    for v in p:
        print(v)







