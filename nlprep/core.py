"""
This sub-module defines core functionality,
like applying or negating filters.
"""
from collections.abc import Callable, Iterable, Iterator

from nlprep.types import Document, Filter, Filter_Result, Pipeline, Pipeline_Generator


def apply_filters(
    docs: Iterable[Document],
    unsafe_filters: Pipeline = tuple(),
    safe_filters: Pipeline = tuple(),
) -> Iterator[Document]:
    """
    Iteratively apply the pipeline's Filter functions on the given documents,
    interpreting the collections of indices returned by the filters as
    tokens to possibly keep.
    Tokens with index not inside a filter's result will be discarded.

    If no filters were given, the documents are returned as-is.

    :param docs: The documents to process.
    :param unsafe_filters: Filters that require the document to be 'intact',
                           i.e. still in the form of natural language.
                           This usually includes filters that act on sentences,
                           POS tags, or require additional per-token context.
    :param safe_filters: Filters that don't require the document to be 'intact'
                         This includes filters that only act on individual
                         tokens, without requiring any context.
    """

    def subset_by_index(doc: Document, *index_sets: Filter_Result) -> Document:
        indices = frozenset.intersection(*index_sets)

        return Document(token for index, token in enumerate(doc) if index in indices)

    def apply_unsafe(doc: Document, *filters: Filter) -> Document:
        """
        Apply all unsafe filters on the original document.
        """
        if len(filters) == 0:
            return doc

        index_sets = [fun(doc) for fun in filters]
        return subset_by_index(doc, *index_sets)

    def apply_safe(doc: Document, *filters: Filter) -> Document:
        """
        Apply each safe filter on the previously filtered document.

        This should result in better performance,
        as the document gets smaller with each application of the filters.
        """
        for fun in filters:
            index_set = fun(doc)
            doc = subset_by_index(doc, index_set)

        return doc

    for doc in docs:
        yield apply_safe(apply_unsafe(doc, *unsafe_filters), *safe_filters)


def apply_pipeline(
    raw_docs: Iterable[str],
    get_pipeline_fun: Pipeline_Generator,
    tokenize_fun: Callable[[str], Document],
    **kwargs,
) -> Iterable[Document]:
    """
    Directly apply the chosen pipeline on completely unprocessed documents.

    Usually, it would be necessary to tokenize the documents first,
    however, this is done automatically using the given tokenize_fun.

    :args kwargs: Additional keyword arguments passed onto the pipeline
                  generator function.
    """
    docs = [tokenize_fun(raw_doc) for raw_doc in raw_docs]
    pipeline = get_pipeline_fun(docs, **kwargs)
    return apply_filters(docs, *pipeline)
