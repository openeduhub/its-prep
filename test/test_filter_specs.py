import test.strategies as wlo_st
from typing import Collection

import hypothesis.strategies as st
import wloprep.specs.filters as filters
from hypothesis import given
from wloprep.types import Document


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
