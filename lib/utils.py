def copy_properties(src, dest):
    for attr in vars(src).keys():
        setattr(dest, attr, getattr(src, attr))