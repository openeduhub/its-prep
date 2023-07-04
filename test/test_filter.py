import test.strategies as wlo_st
from collections.abc import Callable, Collection
from typing import Any

import hypothesis.strategies as st
import wloprep.filter as filt
from hypothesis import given
from wloprep.types import Document, Filter


def nest(fun: Callable[[Any], Any], n: int):
    """Helper function to apply a function onto itself n times."""

    def nested_fun(x):
        for _ in range(n):
            x = fun(x)

        return x

    return nested_fun


@given(wlo_st.documents_with_filters(), st.integers(min_value=1, max_value=3))
def test_exclude_does_negate(
    given_input: tuple[Document, Filter, frozenset[int]], negation_count: int
):
    doc, filter_fun, include_indices = given_input

    # negate the filter an odd amount of times
    negated_filter = nest(filt.exclude, negation_count * 2 - 1)(filter_fun)

    doc_indices = set(range(len(doc)))
    exclude_indices = set(negated_filter(doc))

    # the negated index set is the complement of the given set
    assert include_indices.intersection(exclude_indices) == set()
    assert include_indices.union(exclude_indices) == doc_indices


@given(wlo_st.documents_with_filters(), st.integers(min_value=1, max_value=3))
def test_exclude_double_negation(
    given_input: tuple[Document, Filter, frozenset[int]], negation_count: int
):
    doc, filter_fun, index_set = given_input

    # negate the filter an even amount of times
    negated_filter = nest(filt.exclude, negation_count * 2)(filter_fun)

    assert index_set == negated_filter(doc)


@given(wlo_st.documents_with_multiple_filters())
def test_apply_filters(
    given_input: tuple[Document, Collection[Filter], Collection[frozenset[int]]]
):
    doc, filter_funs, index_sets = given_input

    results = filt.apply_filters(doc, filters=filter_funs)

    collected_indices = set().union(*index_sets)
    expected_results = [
        [token for index, token in enumerate(doc) if index in collected_indices]
    ]

    for result, expected_result in zip(results, expected_results):
        for token in expected_result:
            assert token in result

        for token in result:
            assert token in expected_result
