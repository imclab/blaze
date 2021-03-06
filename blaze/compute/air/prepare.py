# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import threading

from . import ir, pipeline, transforms

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

passes = [
    transforms.explicit_coercions,
]

#------------------------------------------------------------------------
# Execution Context
#------------------------------------------------------------------------

_tls = threading.local()

def current_ctx():
    """Return the current evaluation strategy"""
    try:
        return _tls.ctx
    except AttributeError:
        _tls.ctx = ir.ExecutionContext()
        return current_ctx()

#------------------------------------------------------------------------
# Prepare
#------------------------------------------------------------------------

def prepare(expr, strategy):
    """
    Prepare a Deferred for interpretation
    """
    graph, expr_ctx = expr
    f, values = ir.from_expr(graph, expr_ctx, current_ctx())

    env = {'strategy': strategy}
    func, env = pipeline.run_pipeline(f, env, passes)

    return func, env
