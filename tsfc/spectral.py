from __future__ import absolute_import, print_function, division
from six.moves import map, zip

from collections import OrderedDict
from functools import partial
from itertools import chain

import numpy

import gem
from gem import Delta, Indexed, index_sum
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import sum_factorise as _sum_factorise
from gem.optimise import unroll_indexsum
from gem.refactorise import ATOMIC, COMPOUND, OTHER, MonomialSum, collect_monomials


def delta_elimination(sum_indices, args, rest):
    """IndexSum-Delta cancellation for monomials."""
    factors = [rest] + list(args)  # construct factors
    sum_indices, factors = _delta_elimination(sum_indices, factors)
    # Destructure factors after cancellation
    rest = factors.pop(0)
    args = factors
    return sum_indices, args, rest


def sum_factorise(sum_indices, args, rest):
    """Optimised monomial product construction through sum factorisation
    with reversed sum indices."""
    sum_indices = list(sum_indices)
    sum_indices.reverse()
    factors = args + (rest,)
    return _sum_factorise(sum_indices, factors)


def Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters):
    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)
    # Integral representation: pair with the set of argument indices
    # and a GEM expression
    argument_indices = set(chain(*argument_multiindices))
    return [(argument_indices,
             index_sum(e, quadrature_multiindex))
            for e in expressions]


def flatten(var_reps):
    # Classifier for argument factorisation
    def classify(argument_indices, expression):
        n = len(argument_indices.intersection(expression.free_indices))
        if n == 0:
            return OTHER
        elif n == 1:
            if isinstance(expression, (Delta, Indexed)):
                return ATOMIC
            else:
                return COMPOUND
        else:
            return COMPOUND

    for variable, reps in var_reps:
        # Destructure representation
        argument_indicez, expressions = zip(*reps)
        # Assert identical argument indices for all integrals
        argument_indices, = set(map(frozenset, argument_indicez))
        # Argument factorise
        classifier = partial(classify, argument_indices)
        for monomial_sum in collect_monomials(expressions, classifier):
            foo = OrderedDict()

            for monomial in monomial_sum:
                for v, s, a, r in unconcatenate(variable, *monomial):
                    foo.setdefault(v, []).append((s, a, r))

            for var, monomial_list in foo.items():
                # Compact MonomialSum after IndexSum-Delta cancellation
                delta_simplified = MonomialSum()
                for monomial in monomial_list:
                    delta_simplified.add(*delta_elimination(*monomial))

                # Yield assignments
                for monomial in delta_simplified:
                    yield (var, sum_factorise(*monomial))


finalise_options = {}


def unconcatenate(variable, sum_indices, args, rest):
    # # Eliminate annoying ComponentTensors
    # expression, = remove_componenttensors([expression])

    for arg in args:
        if isinstance(arg, gem.Indexed) and isinstance(arg.children[0], gem.Concatenate):
            index, = arg.multiindex
            if index not in sum_indices:
                break
    else:
        # No Concatenate to handle
        if any(isinstance(node, gem.Concatenate) for node in gem.node.traversal(args + (rest,))):
            raise RuntimeError("Concatenate node cannot be split")

        # Nothing left to do
        return iter(((variable, sum_indices, args, rest),))

    # 'arg' is an indexed Concatenate factor, 'index' is its index
    concat_ref, = arg.children

    split_assignments = []
    offset = 0
    for i, child in enumerate(concat_ref.children):
        size = numpy.prod(child.shape, dtype=int)
        slice_ = slice(offset, offset + size)
        multiindex = tuple(gem.Index(extent=d) for d in child.shape)

        split_args = []
        for arg in args:
            if index not in arg.free_indices:
                split_args.append(args)
                continue

            if isinstance(arg, gem.Indexed) and isinstance(arg.children[0], gem.Concatenate):
                assert arg.multiindex == (index,)
                concat, = arg.children
                section = gem.Indexed(concat.children[i], multiindex)
                section, = gem.optimise.remove_componenttensors([section])
                si, factors = gem.optimise.traverse_product(section)
                assert not si
                split_args.extend(factors)
            # elif isinstance(f, FlexiblyIndexed) and isinstance(f.children[0], Variable):
            #     assert len(f.free_indices) == 1
            #     data = ComponentTensor(f, (index,))
            #     split_factors.append(Indexed(reshape(view(data, slice_), child.shape), multiindex))
            else:
                assert False

        # split_sum_indices = list(sum_indices)
        # split_sum_indices.remove(index)
        # split_sum_indices.extend(multiindex)

        assert index not in rest.free_indices

        assert isinstance(variable, gem.gem.FlexiblyIndexed) and isinstance(variable.children[0], gem.Variable)
        assert len(variable.free_indices) == 1
        data = gem.ComponentTensor(variable, (index,))

        split_variable = gem.Indexed(gem.reshape(gem.view(data, slice_), child.shape), multiindex)
        split_variable, = gem.optimise.remove_componenttensors([split_variable])

        split_assignments.append((split_variable, sum_indices, tuple(split_args), rest))

        offset += size

    return chain.from_iterable(unconcatenate(*assignment) for assignment in split_assignments)
