class BasketIterable(object):
    """
    this class is a generates a well known sequence of numbers
    """
    def __init__(self, items, bunch_size):
        self.items = items
        self.bunch_size = bunch_size
        self.length = len(items)
        self.bunch_number = 0
        self.max_bunch_number = self.length // bunch_size if self.length % bunch_size == 0 else self.length // bunch_size + 1

    def __iter__(self):
        return self    # because the object is both the iterable and the itorator

    def __len__(self):
        return self.length

    def __next__(self):
        if self.bunch_number >= self.max_bunch_number:
            raise StopIteration
        start = self.bunch_number * self.bunch_size
        stop = (self.bunch_number + 1) * self.bunch_size
        stop = self.length if stop > self.length else stop
        self.bunch_number += 1
        return self.items[start:stop]



