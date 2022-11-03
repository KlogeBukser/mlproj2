import numpy as np

def get_2D_pols(deg):
    """ (From project 1)
    Produces every polynomial of two variables up to a given degree

    :deg: int, Degrees
    :returns: array-like, lambda functions

    """
    pols = []
    for j in range(deg + 1):
        # Makes all 2D polynomials up to the given degree
        pols += [(lambda xy, i = i, j = j: xy[0]**(j - i) * xy[1]**i) for i in range(j + 1)]
    return pols

def get_1D_pols(deg):
    """ 
    Produces every polynomial of one variables up to a given degree

    :deg: int, Degrees
    :returns: array-like, lambda functions

    """
    pols = [(lambda x, i = i: x**i) for i in range(deg)]
    return pols

