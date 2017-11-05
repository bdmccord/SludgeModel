
class Counter(object):

    def __init__(self):
        self.count = 0

    def countable(self, func):
        def inner_func(*args, **kwargs):
            self.count += 1
            func(*args, **kwargs)
        return inner_func

    def __repr__(self):
        return "{0} -> count: {1}".format(self.__class__.__name__, self.count)


counter = Counter()

@counter.countable
def add(a, b):
    return a + b

@counter.countable
def mult(a, b):
    return a * b

print(counter)
add(1,2)
print(counter)
mult(1,2)
print(counter)
add(1,2)
print(counter)
