import test.strategies as wlo_st
from collections.abc import Collection, Callable
from typing import TypeVar

import hypothesis.strategies as st
import nlprep.specs.filters as filters
from hypothesis import given
from nlprep.types import Document, Filter, Filter_Result

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
    given_input: tuple[Document, Filter, Filter_Result], negation_count: int
):
    """
    Ensure that the negated filter actually represents the complement
    of the original filter's result
    """
    doc, filter_fun, include_indices = given_input
    # negate the filter an odd number of times
    negated_filter = _nest(filters.negated, negation_count * 2 - 1)(filter_fun)

    doc_indices = set(range(len(doc)))
    exclude_indices = set(negated_filter(doc))

    assert set() == include_indices.intersection(exclude_indices)
    assert doc_indices == include_indices.union(exclude_indices)


@given(wlo_st.documents_with_filters(), st.integers(min_value=1, max_value=3))
def test_negate_double_negation(
    given_input: tuple[Document, Filter, Filter_Result], negation_count: int
):
    """
    Ensure that negating a filter an even number of times
    does not affect its result.
    """
    doc, filter_fun, index_set = given_input
    # negate the filter an even number of times
    negated_filter = _nest(filters.negated, negation_count * 2)(filter_fun)

    assert index_set == negated_filter(doc)


@given(
    wlo_st.documents(),
    st.tuples(st.integers(), st.integers()).map(sorted),
    st.booleans(),
)
def test_get_words_by_df_in_interval_with_both_nums(
    docs: Collection[Document], interval: tuple[int, int], interval_open: bool
):
    min_num, max_num = interval

    result = filters.get_props_by_document_frequency(
        docs,
        property_fun=lambda doc: doc,
        min_num=min_num,
        max_num=max_num,
        interval_open=interval_open,
    )

    vocabulary = set().union(*docs)
    for token in vocabulary:
        count = sum(token in doc for doc in docs)

        if interval_open:
            if min_num < count < max_num:
                assert token in result

        else:
            if min_num <= count <= max_num:
                assert token in result


@given(
    wlo_st.documents(),
    st.integers(),
    st.booleans(),
)
def test_get_words_by_df_in_interval_with_min_num(
    docs: Collection[Document], min_num, interval_open: bool
):
    result = filters.get_props_by_document_frequency(
        docs,
        property_fun=lambda doc: doc,
        min_num=min_num,
        interval_open=interval_open,
    )

    vocabulary = set().union(*docs)
    for token in vocabulary:
        count = sum(token in doc for doc in docs)

        if interval_open:
            if min_num < count:
                assert token in result

        else:
            if min_num <= count:
                assert token in result


@given(
    wlo_st.documents(),
    st.integers(),
    st.booleans(),
)
def test_get_words_by_df_in_interval_with_max_num(
    docs: Collection[Document], max_num, interval_open: bool
):
    result = filters.get_props_by_document_frequency(
        docs,
        property_fun=lambda doc: doc,
        max_num=max_num,
        interval_open=interval_open,
    )

    vocabulary = set().union(*docs)
    for token in vocabulary:
        count = sum(token in doc for doc in docs)

        if interval_open:
            if max_num > count:
                assert token in result

        else:
            if max_num >= count:
                assert token in result


@given(
    wlo_st.documents(),
    st.tuples(
        st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0)
    ).map(sorted),
    st.booleans(),
)
def test_get_words_by_df_in_interval_with_both_rates(
    docs: Collection[Document], interval: tuple[float, float], interval_open: bool
):
    min_rate, max_rate = interval
    min_num, max_num = tuple(len(docs) * rate for rate in interval)

    result = filters.get_props_by_document_frequency(
        docs,
        property_fun=lambda doc: doc,
        min_rate=min_rate,
        max_rate=max_rate,
        interval_open=interval_open,
    )

    vocabulary = set().union(*docs)
    for token in vocabulary:
        count = sum(token in doc for doc in docs)

        if interval_open:
            if min_num < count < max_num:
                assert token in result

        else:
            if min_num <= count <= max_num:
                assert token in result
