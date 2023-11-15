import test.strategies as lanst

from hypothesis import given
from hypothesis import strategies as st
from nlprep.core import apply_filters, selected_properties
from nlprep.types import Document, Filter, Property_Function


@given(st.lists(lanst.documents), st.lists(lanst.filters()))
def test_apply_filters_safe(docs: list[Document], filter_funs: list[Filter]):
    results = apply_filters(docs, filter_funs)

    # if no filters were given, the documents should be unmodified
    if len(filter_funs) == 0:
        for doc, result in zip(docs, results):
            assert doc == result

        return

    # assert that the result is a sub-document
    # and that all tokens to discard have been discarded
    for doc, result in zip(docs, results):
        assert result.selected.issubset(doc.selected)

        for fun in filter_funs:
            discarded_indices = set(range(len(doc.original_tokens))) - fun(doc).selected
            for discarded_index in discarded_indices:
                assert discarded_index not in result.selected


@given(st.lists(lanst.documents), st.lists(lanst.filters_unsafe()))
def test_apply_filters_unsafe(docs: list[Document], filter_funs: list[Filter]):
    results = apply_filters(docs, filter_funs)

    # if no filters were given, the documents should be unmodified
    if len(filter_funs) == 0:
        for doc, result in zip(docs, results):
            assert doc == result

        return

    # assert that the result is a sub-document
    # and that all tokens to discard have been discarded
    # note that here, we use filter functions that may return different
    # sub-documents, depending on the previously selected tokens
    for doc, result in zip(docs, results):
        assert result.selected.issubset(doc.selected)

        for fun in filter_funs:
            doc = fun(doc)
            discarded_indices = set(range(len(doc.original_tokens))) - doc.selected

            for discarded_index in discarded_indices:
                assert discarded_index not in result.selected


@given(st.lists(lanst.documents_with_selections()), lanst.property_funs())
def test_selected_properties(docs: list[Document], property_fun: Property_Function):
    for props, doc in zip(selected_properties(docs, property_fun), docs):
        assert len(props) == len(doc.selected)
