def each_succ_from_to(begin, end):
    x = begin
    while x != end:
        yield x
        x += 1


def each_succ_from(begin):
    for x in each_succ_from_to(begin, None):
        yield x


def each_item(xx):
    for x in xx:
        yield x
