"""
This sub-module defines core functionality,
like applying or negating filters.
"""
from collections.abc import Callable, Iterable, Iterator

from nlprep.types import Document, Filter, Pipeline, Pipeline_Generator


def apply_filters(docs: Iterable[Document], filters: Pipeline) -> Iterator[Document]:
    """
    Iteratively apply the pipeline's Filter functions on the given documents,
    interpreting the collections of indices returned by the filters as
    tokens to possibly keep.
    Tokens with index not inside a filter's result will be discarded.

    If no filters were given, the documents are returned as-is.
    """

    def apply_all(doc: Document, *filters: Filter) -> Document:
        """
        Apply each filter on the previously filtered document.

        Depending on the implementation of the filter,
        this can result in better performance,
        as the document gets smaller with each application of the filters.
        """
        for fun in filters:
            doc = fun(doc)

        return doc

    for doc in docs:
        yield apply_all(doc, *filters)


def tokenize_documents(
    raw_docs: Iterable[str],
    tokenize_fun: Callable[[str], tuple[str, ...]],
) -> Iterator[Document]:
    """Create Document objects from raw text, using the given tokenizer."""
    for raw_doc in raw_docs:
        yield Document.fromtext(raw_doc, tokenize_fun=tokenize_fun)
