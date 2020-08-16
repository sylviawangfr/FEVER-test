class BasketIterable:
    """
    this class is a generates a well known sequence of numbers
    """
    def __init__(self, items, bunch_size):
        self.items = items
        self.bunch_size = bunch_size
        self.length = len(items)
        self.max_bunch_number = self.length // bunch_size if self.length % bunch_size == 0 else self.length // bunch_size + 1

    def __iter__(self):
        return self    # because the object is both the iterable and the itorator

    # def __len__(self):
    #     return self.length

    def __next__(self):
        if len(self.items) < 1:
            raise StopIteration
        bunch = self.bunch_size if len(self.items) >= self.bunch_size else len(self.items)
        re = [self.items.pop(0) for i in range(bunch)]
        return re



