"""
This sub-module defines core functionality related to filtering functions.

Filtering functions represent the individual steps taken
in a pre-processing pipeline.
"""
from collections.abc import Collection
from typing import Protocol

import spacy.tokens

import wloprep.utils as utils


class Filter(Protocol):
    """
    Functions that filter document contents in some way.

    Filter functions are composed to created pipelines.
    """

    def __call__(self, doc: Collection[spacy.tokens.Token]) -> Collection[int]:
        """
        Given a document, return a Collection of indices to keep.

        :param doc: The document to process.
        """
        ...


def apply_filters(
    filters: Collection[Filter],
    *docs: Collection[spacy.tokens.Token],
) -> Collection[Collection[spacy.tokens.Token]]:
    """
    Iteratively apply the pipeline's Filter functions on the given documents,
    interpreting the collections of indices returned by the filters as
    tokens to keep.

    To improve efficiency, each subsequent Filter function's results
    are immediately used to subset the document and then pass that subset
    onto the next filter.
    Thus, the fastest and most impactful functions should be specified first.

    :param docs: The documents to process.
    :param filters: The sequency of Filter functions to apply,
                    Interpreting their resulting indices as tokens
                    to keep.
    """

    def apply_on(
        doc: Collection[spacy.tokens.Token],
    ) -> Collection[spacy.tokens.Token]:
        current_doc = doc
        for fun in filters:
            indices = fun(current_doc)
            current_doc = utils.subset_by_index(current_doc, indices)

        return current_doc

    return [apply_on(doc) for doc in docs]


def exclude(fun: Filter) -> Filter:
    """
    Return a new filter function that returns the negated index collection.

    :param fun: The original filter function to negate.
    """

    def negated_fun(doc: Collection[spacy.tokens.Token]) -> Collection[int]:
        original_indices = set(range(len(doc)))
        ignored_indices = set(fun(doc))

        return original_indices - ignored_indices

    return negated_fun
