import test.strategies as lanst
from collections.abc import Collection, Callable

import hypothesis.strategies as st
import nlprep.specs.filters as filters
from hypothesis import given
from nlprep.types import Document, Filter
from nlprep.utils import nest


@given(
    lanst.documents,
    lanst.filters(),
    st.integers(min_value=1, max_value=4).map(lambda x: 2 * x - 1),
)
def test_negate_does_negate(doc: Document, filter_fun: Filter, negation_count: int):
    """
    Ensure that the negated filter actually represents the complement
    of the original filter's result
    """
    # negate the filter an odd number of times
    negated_filter = nest(filters.negated, negation_count)(filter_fun)

    neg_result = negated_filter(doc)
    pos_result = filter_fun(doc)

    if len(doc.selected_tokens) > 0:
        assert neg_result != pos_result

    assert neg_result.selected & pos_result.selected == set()
    assert neg_result.selected | pos_result.selected == doc.selected


@given(
    lanst.documents,
    lanst.filters(),
    st.integers(min_value=1, max_value=4).map(lambda x: 2 * x),
)
def test_negate_double_negation(doc: Document, filter_fun: Filter, negation_count: int):
    """
    Ensure that the negated filter actually represents the complement
    of the original filter's result
    """
    # negate the filter an even number of times
    negated_filter = nest(filters.negated, negation_count)(filter_fun)

    neg_result = negated_filter(doc)
    pos_result = filter_fun(doc)

    assert neg_result == pos_result


@given(
    st.lists(lanst.documents),
    lanst.property_funs,
    st.tuples(st.integers(), st.integers()).map(sorted),
    st.booleans(),
)
def test_get_words_by_df_in_interval_with_both_nums(
    docs: Collection[Document],
    property_fun: Callable[[Document], tuple[str, ...]],
    interval: tuple[int, int],
    interval_open: bool,
):
    min_num, max_num = interval

    result = filters.get_props_by_document_frequency(
        docs,
        property_fun=property_fun,
        min_num=min_num,
        max_num=max_num,
        interval_open=interval_open,
    )

    property_vocabulary = set().union(*[property_fun(doc) for doc in docs])
    for prop in property_vocabulary:
        props_of_docs = [set(property_fun(doc)) for doc in docs]
        count = sum(prop in props for props in props_of_docs)

        if interval_open:
            if min_num < count < max_num:
                assert prop in result
            else:
                assert prop not in result

        else:
            if min_num <= count <= max_num:
                assert prop in result
            else:
                assert prop not in result


@given(
    st.lists(lanst.documents),
    lanst.property_funs,
    st.integers(),
    st.booleans(),
)
def test_get_words_by_df_in_interval_with_min_num(
    docs: Collection[Document],
    property_fun: Callable[[Document], tuple[str, ...]],
    min_num: int,
    interval_open: bool,
):
    result = filters.get_props_by_document_frequency(
        docs,
        property_fun=property_fun,
        min_num=min_num,
        interval_open=interval_open,
    )

    property_vocabulary = set().union(*[property_fun(doc) for doc in docs])
    for prop in property_vocabulary:
        props_of_docs = [set(property_fun(doc)) for doc in docs]
        count = sum(prop in props for props in props_of_docs)

        if interval_open:
            if min_num < count:
                assert prop in result
            else:
                assert prop not in result

        else:
            if min_num <= count:
                assert prop in result
            else:
                assert prop not in result


@given(
    st.lists(lanst.documents),
    lanst.property_funs,
    st.integers(),
    st.booleans(),
)
def test_get_words_by_df_in_interval_with_max_num(
    docs: Collection[Document],
    property_fun: Callable[[Document], tuple[str, ...]],
    max_num: int,
    interval_open: bool,
):
    result = filters.get_props_by_document_frequency(
        docs,
        property_fun=property_fun,
        max_num=max_num,
        interval_open=interval_open,
    )

    property_vocabulary = set().union(*[property_fun(doc) for doc in docs])
    for prop in property_vocabulary:
        props_of_docs = [set(property_fun(doc)) for doc in docs]
        count = sum(prop in props for props in props_of_docs)

        if interval_open:
            if count < max_num:
                assert prop in result
            else:
                assert prop not in result

        else:
            if count <= max_num:
                assert prop in result
            else:
                assert prop not in result


@given(
    st.lists(lanst.documents),
    lanst.property_funs,
    st.tuples(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    ).map(sorted),
    st.booleans(),
)
def test_get_words_by_df_in_interval_with_both_rates(
    docs: Collection[Document],
    property_fun: Callable[[Document], tuple[str, ...]],
    interval: tuple[float, float],
    interval_open: bool,
):
    min_rate, max_rate = interval

    result = filters.get_props_by_document_frequency(
        docs,
        property_fun=property_fun,
        min_rate=min_rate,
        max_rate=max_rate,
        interval_open=interval_open,
    )

    property_vocabulary = set().union(*[property_fun(doc) for doc in docs])
    for prop in property_vocabulary:
        props_of_docs = [set(property_fun(doc)) for doc in docs]
        rate = sum(prop in props for props in props_of_docs) / len(docs)

        if interval_open:
            if min_rate < rate < max_rate:
                assert prop in result
            else:
                assert prop not in result

        else:
            if min_rate <= rate <= max_rate:
                assert prop in result
            else:
                assert prop not in result
