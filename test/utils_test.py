from utils.common import iter_baskets_contiguous, thread_exe
import unittest
from memory_profiler import profile
from utils.iter_basket import BasketIterable
import gc

class TestUtils(unittest.TestCase):
    def testBasket(self):
        data = range(500 * 500)
        o = BasketIterable(data, 50000)
        for i in o:
            print(len(i))
            gc.collect()

    def test_multithread(self):
        thread_exe(print, iter_baskets_contiguous(range(14), 3), 5, "testing")
        print("done")

    def test_basket(self):
        items = range(5)
        for n in iter_baskets_contiguous(items, 2):
            print(n)
            # assert
        for n in iter_baskets_contiguous(items, 5):
            print(n)
        for n in iter_baskets_contiguous(items, 3):
            print(n)
        for n in iter_baskets_contiguous(items, 6):
            print(n)