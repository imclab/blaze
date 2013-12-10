# -*- coding: utf-8 -*-

"""
SciDB array constructors.
"""

from __future__ import print_function, division, absolute_import

import blaze
from blaze.datashape import from_numpy

from .query import Query, build
from .datatypes import scidb_dshape
from .datadesc import SciDBDataDesc

#------------------------------------------------------------------------
# Array creation
#------------------------------------------------------------------------

def _create(dshape, n, conn, chunk_size=1024, overlap=0):
    sdshape = scidb_dshape(dshape, chunk_size, overlap)
    query = build(sdshape, n)
    return blaze.Array(SciDBDataDesc(dshape, query, conn))

#------------------------------------------------------------------------
# Constructors
#------------------------------------------------------------------------

def empty(dshape, conn, chunk_size=1024, overlap=0):
    """Create an empty array"""
    return zeros(dshape, conn, chunk_size, overlap)

def zeros(dshape, conn, chunk_size=1024, overlap=0):
    """Create an array of zeros"""
    return _create(dshape, "0", conn, chunk_size, overlap)

def ones(dshape, conn, chunk_size=1024, overlap=0):
    """Create an array of ones"""
    return _create(dshape, "1", conn, chunk_size, overlap)

def handle(conn, arrname):
    """Obtain an array handle to an existing SciDB array"""
    scidbpy_arr = conn.wrap_array(arrname)
    dshape = from_numpy(scidbpy_arr.shape, scidbpy_arr.dtype)
    return SciDBDataDesc(dshape, Query(arrname, (), {}), conn)


###### The below is adapted from scidb-py ######

def randint(self, shape, dtype='uint32', lower=0, #upper=SCIDB_RAND_MAX,
            **kwargs):
    """Return an array of random integers between lower and upper

    Parameters
    ----------
    shape: tuple or int
        The shape of the array
    dtype: string or list
        The data type of the array
    lower: float
        The lower bound of the random sample (default=0)
    upper: float
        The upper bound of the random sample (default=2147483647)
    **kwargs:
        Additional keyword arguments are passed to SciDBDataShape.

    Returns
    -------
    arr: SciDBArray
        A SciDBArray consisting of random integers, uniformly distributed
        between `lower` and `upper`.
    """
    arr = self.new_array(shape, dtype, **kwargs)
    fill_value = 'random() % {0} + {1}'.format(upper - lower, lower)
    self.query('store(build({0}, {1}), {0})', arr, fill_value)
    return arr

def arange(self, start, stop=None, step=1, dtype=None, **kwargs):
    """arange([start,] stop[, step,], dtype=None, **kwargs)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the behavior is equivalent to the Python
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use ``linspace`` for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified, `start` must also be
        given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, it is
        inferred from the type of the input arguments.
    **kwargs :
        Additional arguments are passed to SciDBDatashape when creating
        the output array.

    Returns
    -------
    arange : SciDBArray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
    """
    if stop is None:
        stop = start
        start = 0

    if dtype is None:
        dtype = np.array(start + stop + step).dtype

    size = int(np.ceil((stop - start) * 1. / step))

    arr = self.new_array(size, dtype, **kwargs)
    self.query("store(build({A}, {start} + {step} * {A.d0}), {A})",
               A=arr, start=start, step=step)
    return arr

def linspace(self, start, stop, num=50,
             endpoint=True, retstep=False, **kwargs):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop` ].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : scalar
        The starting value of the sequence.
    stop : scalar
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of
        ``num + 1`` evenly spaced samples, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    **kwargs :
        additional keyword arguments are passed to SciDBDataShape

    Returns
    -------
    samples : SciDBArray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float (only if `retstep` is True)
        Size of spacing between samples.
    """
    num = int(num)

    if endpoint:
        step = (stop - start) * 1. / (num - 1)
    else:
        step = (stop - start) * 1. / num

    arr = self.new_array(num, **kwargs)
    self.query("store(build({A}, {start} + {step} * {A.d0}), {A})",
               A=arr, start=start, step=step)

    if retstep:
        return arr, step
    else:
        return arr

def identity(self, n, dtype='double', sparse=False, **kwargs):
    """Return a 2-dimensional square identity matrix of size n

    Parameters
    ----------
    n : integer
        the number of rows and columns in the matrix
    dtype: string or list
        The data type of the array
    sparse: boolean
        specify whether to create a sparse array (default=False)
    **kwargs:
        Additional keyword arguments are passed to SciDBDataShape.

    Returns
    -------
    arr: SciDBArray
        A SciDBArray containint an [n x n] identity matrix
    """
    arr = self.new_array((n, n), dtype, **kwargs)
    if sparse:
        query = 'store(build_sparse({A},1,{A.d0}={A.d1}),{A})'
    else:
        query = 'store(build({A},iif({A.d0}={A.d1},1,0)),{A})'
    self.query(query, A=arr)
    return arr
