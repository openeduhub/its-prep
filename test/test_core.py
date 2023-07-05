import test.strategies as wlo_st
from collections.abc import Callable, Collection
from typing import TypeVar

import hypothesis.strategies as st
from hypothesis import given
from nlprep.core import apply_filters, negated
from nlprep.types import Document, Filter

T = TypeVar("T")


def _nest(fun: Callable[[T], T], n: int) -> Callable[[T], T]:
    """Helper function to apply a function onto itself n times."""

    def nested_fun(x: T) -> T:
        for _ in range(n):
            x = fun(x)

        return x

    return nested_fun


@given(wlo_st.documents_with_filters(), st.integers(min_value=1, max_value=3))
def test_negate_does_negate(
    given_input: tuple[Document, Filter, frozenset[int]], negation_count: int
):
    doc, filter_fun, include_indices = given_input

    # negate the filter an odd amount of times
    negated_filter = _nest(negated, negation_count * 2 - 1)(filter_fun)

    doc_indices = set(range(len(doc)))
    exclude_indices = set(negated_filter(doc))

    # the negated index set is the complement of the given set
    assert include_indices.intersection(exclude_indices) == set()
    assert include_indices.union(exclude_indices) == doc_indices


@given(wlo_st.documents_with_filters(), st.integers(min_value=1, max_value=3))
def test_negate_double_negation(
    given_input: tuple[Document, Filter, frozenset[int]], negation_count: int
):
    doc, filter_fun, index_set = given_input

    # negate the filter an even amount of times
    negated_filter = _nest(negated, negation_count * 2)(filter_fun)

    assert index_set == negated_filter(doc)


@given(wlo_st.documents_with_multiple_filters())
def test_apply_filters(
    given_input: tuple[Document, Collection[Filter], Collection[frozenset[int]]]
):
    doc, filter_funs, index_sets = given_input

    results = apply_filters(doc, unsafe_filters=filter_funs)

    collected_indices = set().union(*index_sets)
    expected_results = [
        Document(token for index, token in enumerate(doc) if index in collected_indices)
    ]

    for result, expected_result in zip(results, expected_results):
        assert result == expected_result
