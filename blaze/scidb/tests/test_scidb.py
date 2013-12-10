# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from blaze import dshape, add, mul, eval
from blaze.scidb import connect, empty, zeros, ones, linalg
from blaze.scidb.tests.mock import MockedConn

from scidbpy import interface, SciDBQueryError, SciDBArray

ds = dshape('10, 10, float64')

class TestSciDB(unittest.TestCase):

    def setUp(self):
        self.conn = MockedConn()

    def test_query(self):
        a = zeros(ds, self.conn)
        b = ones(ds, self.conn)

        expr = add(a, mul(a, b))

        graph, ctx = expr.expr
        self.assertEqual(graph.dshape, dshape('10, 10, float64'))

        result = eval(expr)

        self.assertEqual(len(self.conn.recorded), 1)
        [(query, persist)] = self.conn.recorded

        query = str(query)

        self.assertIn("+", query)
        self.assertIn("*", query)
        self.assertIn("build", query)

    def test_query_exec(self):
        print("establishing connection...")
        conn = interface.SciDBShimInterface('http://192.168.56.101:8080/')
        print(conn)

        a = zeros(ds, conn)
        b = ones(ds, conn)

        expr = a + b

        graph, ctx = expr.expr
        self.assertEqual(graph.dshape, dshape('10, 10, float64'))

        result = eval(expr)
        print(result)

    def test_svd(self):
        print("establishing connection...")
        conn = interface.SciDBShimInterface('http://192.168.56.101:8080/')
        print(conn)

        a = zeros(ds, conn)
        b = ones(ds, conn)

        print(linalg.svd(a + b))


if __name__ == '__main__':
    #unittest.main()
    TestSciDB('test_svd').debug()
