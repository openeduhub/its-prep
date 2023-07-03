"""
This sub-module contains various utility functions used across this project,
without necessarily belonging to any particular sub-module directly.
"""
from collections.abc import Collection, Iterable
from typing import Optional

from spacy.tokens import Doc


def subset_by_index(it: Iterable, *index_cols: Collection[int]) -> Collection:
    """
    Return the subset of the given iterable by keeping only wanted indices.

    I.e. ignore only entries with indices
    that are not inside any of the given index collections.

    :param col: The iterable to subset.
    :param index_cols: The collections of indices to retain.
    """
    indices = set().union(*index_cols)

    return [entry for index, entry in enumerate(it) if index in indices]


def get_document_freqs(docs: Collection[Collection[str]]) -> dict[str, int]:
    """
    Return a dictionary mapping words to their document frequency.

    This is particularly useful when filtering out tokens according to their
    document frequency (i.e. through get_tokens_by_df_outside_interval).

    :param docs: The document corpus to process.
    """
    docs_as_sets = [{word for word in doc} for doc in docs]
    vocabulary = set().union(*docs_as_sets)
    return {
        word: sum([word in document for document in docs_as_sets])
        for word in vocabulary
    }


def get_words_by_df_in_interval(
    *docs: Doc,
    min_num: Optional[int] = None,
    max_num: Optional[int] = None,
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None,
    interval_open: bool = False,
    lemmatize: bool = True
) -> Collection[str]:
    """
    Return the words with document frequency outside the given interval.

    Directions that are not given are considered to be unbounded.

    This can be used to remove tokens that are too rare to reason about
    or too frequent to carry much meaning.

    :param docs: The document corpus to process.
    :param min_num: The lower bound of the interval, as the absolute number.
    :param max_num: The upper bound of the interval, as the absolute number.
    :param min_rate: The lower bound of the interval, as the relative rate.
    :param max_rate: The upper bound of the interval, as the relative rate.
    :param interval_open: Consider the interval to be open,
                          i.e. do not include words exactly at the boundaries.
    :param lemmatize: Whether to use the lemmatized versions of the tokens
                      during calculation of the document frequency.
    """
    # override the interval boundaries according to the given rates
    if min_rate:
        min_num = int(len(docs) * min_rate)

    if max_rate:
        max_num = int(len(docs) * max_rate)

    # helper functions to determine whether a count is in the interval
    def right_of_min(count):
        if min_num is None:
            return True

        if interval_open:
            return count < min_num
        else:
            return count <= min_num

    def left_of_max(count):
        if max_num is None:
            return True

        if interval_open:
            return count > max_num
        else:
            return count >= max_num

    def in_interval(count):
        return right_of_min(count) and left_of_max(count)

    dfs = get_document_freqs(
        [[(token.lemma_ if lemmatize else token.text) for token in doc] for doc in docs]
    )
    return {
        (token.lemma_ if lemmatize else token.text)
        for doc in docs
        for token in doc
        if in_interval(dfs[(token.lemma_ if lemmatize else token.text)])
    }
