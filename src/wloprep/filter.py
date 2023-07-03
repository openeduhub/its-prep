"""
This sub-module defines core functionality related to filtering functions.

Filtering functions represent the individual steps taken
in a pre-processing pipeline.
"""
from collections.abc import Collection
from typing import Protocol

from spacy.tokens import Doc, Token

import wloprep.utils as utils


class Filter(Protocol):
    """
    Functions that filter document contents in some way.

    Filter functions are composed to created pipelines.
    """

    def __call__(self, doc: Doc) -> Collection[int]:
        """
        Given a document, return a Collection of indices to keep.

        :param doc: The document to process.
        """
        ...


def apply_filters(
    filters: Collection[Filter],
    *docs: Doc,
) -> Collection[Collection[Token]]:
    """
    Iteratively apply the pipeline's Filter functions on the given documents,
    interpreting the collections of indices returned by the filters as
    tokens to keep.

    :param docs: The documents to process.
    :param filters: The sequency of Filter functions to apply,
                    Interpreting their resulting indices as tokens
                    to keep.
    """

    def apply_on(
        doc: Doc,
    ) -> Collection[Token]:
        indices = [fun(doc) for fun in filters]
        return utils.subset_by_index(doc, *indices)

    return [apply_on(doc) for doc in docs]


def exclude(fun: Filter) -> Filter:
    """
    Return a new filter function that returns the negated index collection.

    :param fun: The original filter function to negate.
    """

    def negated_fun(doc: Doc) -> Collection[int]:
        original_indices = set(range(len(doc)))
        ignored_indices = set(fun(doc))

        return original_indices - ignored_indices

    return negated_fun
