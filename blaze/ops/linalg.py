# -*- coding: utf-8 -*-

"""
Blaze linear algebra python implementations.
"""

from __future__ import print_function, division, absolute_import
import blaze
from blaze.function import function

import numpy as np

__all__ = ['svd']

def svd(A, return_U=True, return_S=True, return_VT=True):
    """Compute the Singular Value Decomposition of the array A:

    A = U.S.V^T

    Parameters
    ----------
    A : SciDBArray
        The array for which the SVD will be computed.  It should be a
        2-dimensional array with a single value per cell.  Currently, the
        svd routine requires non-overlapping chunks of size 32.
    return_U, return_S, return_VT : boolean
        if any is True, then return the associated array.  All are True
        by default

    Returns
    -------
    [U], [S], [VT] : SciDBArrays
        Arrays storing the singular values and vectors of A.
    """
    return np.linalg.svd(A)
