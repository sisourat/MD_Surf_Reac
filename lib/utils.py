

evtoau = 1.0/27.211324570273
angtoau = 1.0/0.5291772109217
fstoau = 1.0/0.02418884326585747
def copy_properties(src, dest):
    for attr in vars(src).keys():
        setattr(dest, attr, getattr(src, attr))