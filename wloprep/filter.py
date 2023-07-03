"""
This sub-module defines core functionality related to filtering functions.

Filtering functions represent the individual steps taken
in a pre-processing pipeline.
"""
from collections.abc import Collection
from typing import Protocol

import wloprep.utils as utils

# On the lowest level, represent documents as collections of tokens.
# These tokens are considered the most atomic part of the document.
Document = tuple[str]


class Filter(Protocol):
    """
    Functions that filter tokenized documents contents in some way.

    Filter functions are composed to created pipelines.
    """

    def __call__(self, doc: Document) -> Collection[int]:
        """
        Given a tokenized document,
        return a Collection of token-indices to keep.

        :param doc: The tokenized document to process.
        """
        ...


def apply_filters(
    filters: Collection[Filter],
    *docs: Document,
) -> Collection[Collection[str]]:
    """
    Iteratively apply the pipeline's Filter functions on the given documents,
    interpreting the collections of indices returned by the filters as
    tokens to keep.

    :param docs: The documents to process.
    :param filters: The sequency of Filter functions to apply,
                    Interpreting their resulting indices as tokens
                    to keep.
    """

    def subset_by_index(doc: Document, *index_cols: Collection[int]) -> Collection:
        indices = set().union(*index_cols)

        return [token for index, token in enumerate(doc) if index in indices]

    def apply_on(
        doc: Document,
    ) -> Collection[str]:
        indices = [fun(doc) for fun in filters]
        return subset_by_index(doc, *indices)

    return [apply_on(doc) for doc in docs]


def exclude(fun: Filter) -> Filter:
    """
    Return a new filter function that returns the negated index collection.

    :param fun: The original filter function to negate.
    """

    def negated_fun(doc: Document) -> Collection[int]:
        original_indices = set(range(len(doc)))
        ignored_indices = set(fun(doc))

        return original_indices - ignored_indices

    return negated_fun
