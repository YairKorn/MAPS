REL_ERROR = 0.01

# Relative error
def float_comprasion(x, y):
    return (abs(x)-abs(y))/(abs(x)) < REL_ERROR