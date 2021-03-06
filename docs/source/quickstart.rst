Quickstart
===========

Blaze Arrays
~~~~~~~~~~~~

.. doctest::

    >>> from blaze import array, dshape
    >>> ds = dshape('2, 2, int32')
    >>> a = array([[1,2],[3,4]], ds)

.. doctest::

    >>> a
    array([[1, 2],
           [3, 4]],
          dshape='2, 2, int32')


Disk Backed Array
~~~~~~~~~~~~~~~~~

.. doctest::

    >>> from blaze import array, dshape, Storage
    >>> p = Storage('blz://foo.blz')
    >>> ds = dshape('2, 2, int32')
    >>> a = array([[1,2],[3,4]], ds, storage=p)


.. doctest::

    >>> a
    array([[1, 2],
           [3, 4]],
          dshape='2, 2, int32')


.. doctest::

    >>> from blaze import open, drop, Storage
    >>> open(Storage('blz://foo.blz'))
    array([[1, 2],
           [3, 4]],
          dshape='2, 2, int32')
    >>> drop(Storage('blz://foo.blz'))
    

Iterators
~~~~~~~~~


.. doctest::

    >>> alst = [1,2,3]
    >>> array(alst.__iter__(), dshape='3, int32')
    array([1, 2, 3],
          dshape='3, int32')



.. XXX: Added a dedicated toplevel page

.. Uncomment this when a way to remove the 'toplevel' from description
.. would be found...
.. Top level functions
.. ~~~~~~~~~~~~~~~~~~~

.. .. automodule blaze.toplevel
..    :members:
