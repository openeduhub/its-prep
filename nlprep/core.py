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
    for raw_doc in raw_docs:
        yield Document.fromtext(raw_doc, tokenize_fun=tokenize_fun)


def apply_pipeline(
    raw_docs: Iterable[str],
    get_pipeline_fun: Pipeline_Generator,
    tokenize_fun: Callable[[str], tuple[str, ...]],
    **kwargs,
) -> Iterable[Document]:
    """
    Directly apply the chosen pipeline on completely unprocessed documents.

    Usually, it would be necessary to tokenize the documents first,
    however, this is done automatically using the given tokenize_fun.

    :args kwargs: Additional keyword arguments passed onto the pipeline
                  generator function.
    """
    docs = [
        Document.fromtext(raw_doc, tokenize_fun=tokenize_fun) for raw_doc in raw_docs
    ]
    pipeline = get_pipeline_fun(docs, **kwargs)
    return apply_filters(docs, pipeline)
