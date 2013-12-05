# -*- coding: utf-8 -*-

"""
Blaze term tree evaluation.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit
from numba2.runtime.obj.tupleobject import head, tail, StaticTuple, EmptyTuple

#===------------------------------------------------------------------===
# Blaze Term and Evaluation Engine
#===------------------------------------------------------------------===

_cache = {} # (f, subterm1, subterm2) -> Appl

def make_applier(f, subterm1, subterm2):
    """Create a blaze function application term"""

    @jit('Apply[arg1, arg2]')
    class Apply(object):
        """Function application term"""
        layout = [('arg1', 'arg1'), ('arg2', 'arg2')]

        @jit
        def apply(self, args):
            arg1 = eval(self.arg1, args)
            arg2 = eval(self.arg2, args)
            return f(arg1, arg2)

        def __str__(self):
            return "Apply[%s, %s, %s]" % (f, self.arg1, self.arg2)

    return Apply(subterm1, subterm2)


@jit('Arg[argno]')
class Arg(object):
    """Argument to a term."""

    layout = [('argno', 'argno')]

    @jit
    def apply(self, args):
        return lookup(args, self.argno)

    def __str__(self):
        return "Arg[%s]" % (self.argno,)


@jit('Succ[pred]')
class Succ(object):
    """Term to represent an index into the arguments tuple"""
    layout = [('pred', 'pred')]

    def __str__(self):
        return "Succ[%s]" % (self.pred,)

# -- eval -- #

@jit('StaticTuple[a, b] -> Succ[pred] -> r')
def lookup(args, n):
    """Look up the nth argument"""
    return lookup(tail(args), n.pred)

@jit('StaticTuple[a, b] -> zero -> r')
def lookup(args, n):
    return head(args)

@jit
def eval(term, args):
    """Evaluate a given term with the argument tuple"""
    return term.apply(args)

#===------------------------------------------------------------------===
# Helpers
#===------------------------------------------------------------------===

def argmarker():
    """Create an stream of argument indices into the args tuple"""
    item = 0
    while True:
        yield Arg(item)
        item = Succ(item)

#===------------------------------------------------------------------===
# test
#===------------------------------------------------------------------===

@jit('a -> a -> a')
def add(a, b):
    return a + b

@jit('a -> a -> a')
def mul(a, b):
    return a * b

makearg = argmarker()

a = next(makearg)
b = next(makearg)
x = make_applier(add, a, b) # a + b
term = make_applier(mul, x, x)
print(term)

print(eval(term, (2, 3)))
print(eval(term, (7, 3)))