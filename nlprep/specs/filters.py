"""
A collection of filter functions used in the pipelines from pipeline_spec.

Because these filter functions are not completely static,
i.e. they should be parameterized first or depend on the input data,
they are not defined directly, but created factory functions instead.

These can be used as-is inside of pipeline definitions,
or as guidance for defining further filtering functions.
"""

from collections.abc import Callable, Collection
from typing import Any, Optional, Set, TypeVar

import numpy as np
from nlprep.types import Document, Filter

T = TypeVar("T")


def negated(fun: Filter) -> Filter:
    """
    Return a new filter function that returns the negated index collection.

    :param fun: The original filter function to negate.
    """

    def negated_fun(doc: Document) -> Document:
        doc_filtered = fun(doc)
        return doc.sub_doc(doc.selected - doc_filtered.selected)

    return negated_fun


def get_filter_by_property(
    property_fun: Callable[[Document], Collection[T]],
    req_properties: Collection[T],
) -> Filter:
    """
    A filter that gets tokens based on an arbitrary property.

    This can be used to remove filter based universal POS tags,
    a particular (un)wanted vocabulary of lemmatized tokens, etc.

    :param property_fun: The function to use to analyze the document,
                         obtaining the property to check for.
    :param req_properties: The collection of required properties.
    """

    def filter_fun(doc: Document) -> Document:
        return doc.sub_doc(
            {
                index
                for index, prop in enumerate(property_fun(doc))
                if prop in req_properties
            }
        )

    return filter_fun


def get_filter_by_bool_fun(bool_fun: Callable[[Document], Collection[bool]]) -> Filter:
    """
    Return a filter that returns the tokens that are considered True.

    This can be used to filter for stop words, for example.

    :param bool_fun: The function to use to analyze the document.
    """

    def filter_fun(doc: Document) -> Document:
        return doc.sub_doc(
            {index for index, is_true in enumerate(bool_fun(doc)) if is_true}
        )

    return filter_fun


def __in_interval(
    x: float, lower: Optional[float], upper: Optional[float], interval_open: bool
) -> bool:
    lower = lower if lower is not None else -np.inf
    upper = upper if upper is not None else np.inf

    if interval_open:
        return lower < x < upper

    return lower <= x <= upper


def get_props_by_document_frequency(
    docs: Collection[Document],
    property_fun: Callable[[Document], Collection[T]],
    min_num: Optional[float] = None,
    max_num: Optional[float] = None,
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None,
    interval_open: bool = False,
) -> Set[T]:
    """
    Return the words where the corresponding property
    has a document frequency within the given interval.

    Directions that are not given are considered to be unbounded.

    :param property_fun: The function to use to analyze the documents,
                         obtaining the property to base the count on.
    :param min_rate: The lower bound of the interval, as the relative rate.
                     Overrides min_num if given.
    :param max_rate: The upper bound of the interval, as the relative rate.
                     Overrides max_num if given.
    :param interval_open: Consider the interval to be open,
                          i.e. do not include words exactly at the boundaries.
    """
    # override the interval boundaries according to the given rates
    if min_rate is not None:
        min_num = len(docs) * min_rate

    if max_rate is not None:
        max_num = len(docs) * max_rate

    # helper function to compute document frequencies
    def get_document_freqs() -> dict[T, int]:
        docs_as_sets = [{prop for prop in property_fun(doc)} for doc in docs]
        vocabulary = set().union(*docs_as_sets)
        return {
            prop: sum([prop in document for document in docs_as_sets])
            for prop in vocabulary
        }

    dfs = get_document_freqs()
    return {
        prop
        for prop, count in dfs.items()
        if __in_interval(count, min_num, max_num, interval_open)
    }


def get_filter_by_frequency(
    docs: Collection[Document],
    property_fun: Callable[[Document], Collection[T]],
    min_num: Optional[int] = None,
    max_num: Optional[int] = None,
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None,
    interval_open: bool = False,
) -> Filter:
    """
    Filter for token properties with document frequency
    inside the given interval.

    This can be used to remove tokens that are too rare to reason about
    or too frequent to carry much meaning.

    See get_words_by_property_frequency for more details.
    """
    props_inside_interval = get_props_by_document_frequency(
        property_fun=property_fun,
        docs=docs,
        min_num=min_num,
        max_num=max_num,
        min_rate=min_rate,
        max_rate=max_rate,
        interval_open=interval_open,
    )

    return get_filter_by_property(
        property_fun=property_fun, req_properties=props_inside_interval
    )


def get_filter_by_subset_len(
    subset_fun: Callable[[Document], Collection[Collection[Any]]],
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    interval_open: bool = False,
) -> Filter:
    """
    Filter tokens based on the length of document subsets they belong to.

    This can be used to filter based on the length of sentences.

    :param subset_fun: The function that splits a document
                       into a collection of subsets.
    """

    def filter_fun(doc: Document) -> Document:
        len_by_token = [len(col) for col in subset_fun(doc) for _ in col]
        return doc.sub_doc(
            {
                index
                for index, length in enumerate(len_by_token)
                if __in_interval(length, min_len, max_len, interval_open)
            }
        )

    return filter_fun
