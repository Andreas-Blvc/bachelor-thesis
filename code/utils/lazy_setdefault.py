def lazy_setdefault(d, key, factory):
    if key not in d:
        d[key] = factory()
    return d[key]