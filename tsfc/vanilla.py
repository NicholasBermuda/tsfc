from __future__ import absolute_import, print_function, division

import gem
from gem import index_sum
from gem.node import Memoizer, reuse_if_untouched
from gem.optimise import unroll_indexsum

import numpy
from singledispatch import singledispatch


@singledispatch
def _replace(node, self):
    raise AssertionError


@_replace.register(gem.Node)
def _replace_node(node, self):
    if node in self.substitutions:
        return self.substitutions[node]
    else:
        return reuse_if_untouched(node, self)


def replace(expression, substitutions):
    mapper = Memoizer(_replace)
    mapper.substitutions = substitutions
    return mapper(expression)


def Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters):
    """Constructs an integral representation for each GEM integrand
    expression.

    :arg expressions: integrand multiplied with quadrature weight;
                      multi-root GEM expression DAG
    :arg quadrature_multiindex: quadrature multiindex (tuple)
    :arg argument_multiindices: tuple of argument multiindices,
                                one multiindex for each argument
    :arg parameters: parameters dictionary

    :returns: list of integral representations
    """
    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)
    # Integral representation: just a GEM expression
    return [index_sum(e, quadrature_multiindex) for e in expressions]


def flatten(var_reps):
    """Flatten mode-specific intermediate representation to a series of
    assignments.

    :arg var_reps: series of (return variable, [integral representation]) pairs

    :returns: series of (return variable, GEM expression root) pairs
    """
    for variable, reps in var_reps:
        expressions = reps  # representations are expressions
        expressions = gem.optimise.remove_componenttensors(expressions)
        for expression in expressions:
            argument_indices = expression.free_indices  # ugly
            nodes = []
            for node in gem.node.traversal([expression]):  # assume pre_traversal
                if isinstance(node, gem.Indexed) and isinstance(node.children[0], gem.Concatenate):
                    index, = node.multiindex
                    if index in argument_indices and (len(nodes) == 0 or {index} == set(n.multiindex[0] for n in nodes)):
                        nodes.append(node)

            if nodes:
                concat_ref, = nodes[0].children
                index, = nodes[0].multiindex
                offset = 0
                for i, child in enumerate(concat_ref.children):
                    size = numpy.prod(child.shape, dtype=int)
                    slice_ = slice(offset, offset + size)
                    multiindex = tuple(gem.Index(extent=d) for d in child.shape)

                    substitutions = {node: gem.Indexed(node.children[0].children[i], multiindex)
                                     for node in nodes}
                    expr = replace(expression, substitutions)

                    assert len(variable.free_indices) == 1
                    data = gem.ComponentTensor(variable, (index,))
                    var = gem.Indexed(gem.reshape(gem.view(data, slice_), child.shape), multiindex)
                    var, = gem.optimise.remove_componenttensors([var])
                    yield (var, expr)

                    offset += size
            else:
                # No Concatenate to handle
                yield (variable, expression)


finalise_options = {}
"""To avoid duplicate work, these options that are safe to pass to
:py:func:`gem.impero_utils.preprocess_gem`."""
