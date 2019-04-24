# this thread executor wrapped ** , it takes in a func and a list of parameters repeated in the func
import concurrent.futures
import time


def thread_exe(func, pieces, thd_num):
    with concurrent.futures.ThreadPoolExecutor(thd_num) as executor:
        to_be_done = {executor.submit(func, param): param for param in pieces}
        for t in concurrent.futures.as_completed(to_be_done):
            to_be_done[t]


def wait_delay(d):
    print("sleeping ", d)
    time.sleep(d)
    print("slept for ", d)


def test():
    thread_exe(wait_delay, range(10), 5)
    print("done")

