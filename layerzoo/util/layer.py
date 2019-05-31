def normalize_coords(x, ndim):
    if isinstance(x, int):
        return (x,) * ndim
    else:
        assert len(x) == ndim
        return x
