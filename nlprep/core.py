"""
This sub-module defines core functionality,
like applying or negating filters.
"""
from collections.abc import Collection, Iterator
from typing import Optional

from nlprep.types import Document, Filter, Filter_Result


def apply_filters(
    *docs: Document,
    safe_filters: Optional[Collection[Filter]] = None,
    unsafe_filters: Optional[Collection[Filter]] = None,
) -> Iterator[Document]:
    """
    Iteratively apply the pipeline's Filter functions on the given documents,
    interpreting the collections of indices returned by the filters as
    tokens to keep.

    :param docs: The documents to process.
    :param unsafe_filters: Filters that require the document to be 'intact',
                           i.e. still in the form of natural language.
                           This usually includes filters that act on sentences,
                           POS tags, or require additional per-token context.
    :param safe_filters: Filters that don't require the document to be 'intact'
                         This includes filters that only act on individual
                         tokens, without requiring any context.
    """
    safe_filters = safe_filters if safe_filters else []
    unsafe_filters = unsafe_filters if unsafe_filters else []

    def subset_by_index(doc: Document, *index_cols: Filter_Result) -> Document:
        indices = set().union(*index_cols)

        return Document(token for index, token in enumerate(doc) if index in indices)

    def apply_unsafe(doc: Document, filters: Collection[Filter]) -> Document:
        """
        Apply all unsafe filters on the original document.
        """
        index_sets = [fun(doc) for fun in filters]
        return subset_by_index(doc, *index_sets)

    def apply_safe(doc: Document, filters: Collection[Filter]) -> Document:
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
        yield apply_safe(apply_unsafe(doc, unsafe_filters), safe_filters)


def negated(fun: Filter) -> Filter:
    """
    Return a new filter function that returns the negated index collection.

    :param fun: The original filter function to negate.
    """

    def negated_fun(doc: Document) -> Filter_Result:
        original_indices = Filter_Result(range(len(doc)))
        ignored_indices = fun(doc)

        return original_indices - ignored_indices

    return negated_fun
