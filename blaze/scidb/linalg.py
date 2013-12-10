# -*- coding: utf-8 -*-

"""
SciDB linear algebra.
"""

from __future__ import print_function, division, absolute_import

import blaze
from blaze.function import function
from blaze.ops.linalg import svd

from .kernel import scidb_function
from .query import qformat, execute_query

import numpy as np

###### The below is adapted from scidb-py ######

def dot(self, A, B):
    """Compute the matrix product of A and B

    Parameters
    ----------
    A : SciDBArray
        A must be a two-dimensional matrix of shape (n, p)
    B : SciDBArray
        B must be a two-dimensional matrix of shape (p, m)

    Returns
    -------
    C : SciDBArray
        The wrapper of the SciDB Array, of shape (n, m), consisting of the
        matrix product of A and B
    """
    # TODO: use GEMM and repartition where applicable.  GEMM requires the
    #       chunk size to be 32.  We should probably not repartition the
    #       arrays silently, but instead raise an efficiency warning and
    #       provide a flag that enables automatic repartitioning.

    # TODO: make matrix-vector and vector-vector cases more efficient.
    #       Currently they involve creating copies of the arrays, but this
    #       is just a place-holder for a more efficient implementation.

    if A.ndim not in (1, 2) or B.ndim not in (1, 2):
        raise ValueError("dot requires 1 or 2-dimensional arrays")

    if A.shape[-1] != B.shape[0]:
        raise ValueError("array dimensions must match for dot product")

    output_shape = A.shape[:-1] + B.shape[1:]

    # TODO: the following four transformations should be done by building
    #       a single query rather than executing separate queries.
    #       The following should be considered a place-holder for right
    #       now.
    if A.ndim == 1:
        A = A.reshape((1, A.size))

    if B.ndim == 1:
        B = B.reshape((B.size, 1))

    if A.sdbtype.nullable[0]:
        A = A.substitute(0)

    if B.sdbtype.nullable[0]:
        B = B.substitute(0)

    C = self.new_array()
    self.query('store(multiply({0},{1}),{2})', A, B, C)

    if C.shape == output_shape:
        return C
    elif len(output_shape) == 0:
        return C[0, 0]
    else:
        return C.reshape(output_shape)

#@scidb_function('a -> b -> c -> d -> e') #('A, A, T -> bool -> bool -> bool -> dummy')
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
    #if (A.dshape.ndim != 2):
    #    raise ValueError("svd requires 2-dimensional arrays")
    A = blaze.eval(A)

    out_dict = dict(U=return_U, S=return_S, VT=return_VT)

    # TODO: check that data type is double and chunk size is 32
    ret = []
    for output in ['U', 'S', 'VT']:
        if out_dict[output]:
            query = qformat("gesvd({0}, '{1}'), {2})", A._data.query, output)
            # TODO: do this once :)
            query.stmts.append("load_library('dense_linear_algebra')")
            ret.append(blaze.eval(query))

    return tuple(ret)


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
    A = blaze.eval(A, strategy='scidb')

    if A.ndim != 2:
        raise ValueError("svd requires 2-dimensional arrays")

    out_dict = dict(U=return_U, S=return_S, VT=return_VT)

    conn = A._data.conn
    execute_query(conn, "load_library('dense_linear_algebra')")

    # TODO: check that data type is double and chunk size is 32
    ret = []
    for i, output in enumerate(['U', 'S', 'VT']):
        if out_dict[output]:
            expr = _svd(A, i)
            result = blaze.eval(expr, strategy='scidb')
            ret.append(result)
    return tuple(ret)

# Dummy python impl
@function('a -> b -> c') #('A, A, T -> int64 -> R')
def _svd(A, kind):
    raise NotImplementedError

# Scidb impl
@scidb_function('a -> b -> c') #('A, A, T -> int64 -> R')
def _svd(A, kind):
    type = ['right', 'left', 'values'][kind]
    return qformat("gesvd({0}, '{1}')", A, type)