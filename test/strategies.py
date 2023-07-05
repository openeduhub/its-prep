from collections.abc import Collection
from nlprep.types import Document, Filter, Filter_Result
import hypothesis.strategies as st


@st.composite
def documents(draw, vocabulary=st.integers(), max_doc_size: int = 100) -> Document:
    doc_as_list = draw(st.lists(vocabulary, max_size=max_doc_size))
    doc = Document(str(token) for token in doc_as_list)
    return doc


@st.composite
def filters(draw, doc: Document) -> tuple[Filter, Filter_Result]:
    """Return a filter function with a valid result on the given document."""
    doc_size = len(doc)

    # document is empty
    if not doc_size:
        return lambda doc: Filter_Result(), Filter_Result()

    filter_fun = draw(
        st.functions(
            like=lambda doc: ...,
            returns=st.frozensets(st.integers(min_value=0, max_value=doc_size - 1)),
            pure=True,
        )
    )

    return filter_fun, filter_fun(doc)


@st.composite
def documents_with_filters(
    draw, vocabulary=st.integers(), max_doc_size: int = 100
) -> tuple[Document, Filter, Filter_Result]:
    doc = draw(documents(vocabulary=vocabulary, max_doc_size=max_doc_size))

    filter_fun, result = draw(filters(doc))

    return doc, filter_fun, result


@st.composite
def documents_with_multiple_filters(
    draw, vocabulary=st.integers(), max_doc_size: int = 100
) -> tuple[Document, Collection[Filter], Collection[Filter_Result]]:
    doc = draw(documents(vocabulary=vocabulary, max_doc_size=max_doc_size))
    filter_tuples = draw(st.lists(filters(doc)))

    filter_funs, results = tuple(
        [filter_tuple[index] for filter_tuple in filter_tuples] for index in range(2)
    )

    return doc, filter_funs, results
