# this thread executor wrapped ** , it takes in a func and a list of parameters repeated in the func
import concurrent.futures
from tqdm import tqdm


def thread_exe(func, pieces, thd_num, description):
    with concurrent.futures.ThreadPoolExecutor(thd_num) as executor:
        to_be_done = {executor.submit(func, param): param for param in pieces}
        for t in tqdm(concurrent.futures.as_completed(to_be_done), total=len(list(pieces)), desc=description, position=0):
            to_be_done[t]


# def wait_delay(d):
#     print(d)
#     d_list = []
#     with open(d, encoding='utf-8', mode='r') as in_f:
#         for line in in_f:
#             item = json.loads(line.strip())
#             d_list.append(item)
#     print(len(d_list))
#
#
# def test():
#     # print(len(list(config.WIKI_PAGE_PATH.iterdir())))
#     thread_exe(wait_delay, config.WIKI_PAGE_PATH.iterdir(), 5, "testing")
#     print("done")






