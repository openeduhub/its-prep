import test.strategies as wlo_st
from collections.abc import Collection

from hypothesis import given
from nlprep.core import apply_filters
from nlprep.types import Document, Filter, Filter_Result


@given(wlo_st.documents_with_multiple_filters())
def test_apply_filters_unsafe(
    given_input: tuple[Document, Collection[Filter], Collection[Filter_Result]]
):
    doc, filter_funs, index_sets = given_input

    results = apply_filters([doc], unsafe_filters=filter_funs)

    # if no filters were given, the document should be unmodified
    if len(filter_funs) == 0:
        assert doc == results.__next__()
        return

    collected_indices = Filter_Result.intersection(*index_sets)
    expected_result = Document(
        token for index, token in enumerate(doc) if index in collected_indices
    )

    assert expected_result == results.__next__()


@given(wlo_st.documents_with_chained_filters())
def test_apply_filters_safe(
    given_input: tuple[Document, Collection[Filter], Collection[Filter_Result]]
):
    doc, filter_funs, index_sets = given_input

    results = apply_filters([doc], safe_filters=filter_funs)

    # if no filters were given, the document should be unmodified
    if len(filter_funs) == 0:
        assert doc == results.__next__()
        return

    expected_result = doc
    for index_set in index_sets:
        expected_result = Document(
            token for index, token in enumerate(expected_result) if index in index_set
        )

    assert expected_result == results.__next__()
