"""
A collection of filter functions used in the pipelines from pipeline_spec.

Because these filter functions are not completely static,
i.e. they should be parameterized first or depend on the input data,
they are not defined directly, but created factory functions instead.

These can be used as-is inside of pipeline definitions,
or as guidance for defining further filtering functions.
"""

from typing import Collection, Optional

import spacy.tokens

import wloprep.utils as utils
from wloprep.filter import Filter


def get_filter_by_upos(upos_tags: Collection[str]) -> Filter:
    """
    Return a filter that returns the tokens depending on their UPOS tag.

    This can be used to remove words that tend to be relevant to the task,
    such as prepositions.

    :param upos_tags: The collection of universal POS tags to look for.
    """

    def filter_fun(doc: Collection[spacy.tokens.Token]) -> Collection[int]:
        return {index for index, token in enumerate(doc) if token.pos_ in upos_tags}

    return filter_fun


def get_filter_by_stop_words(
    additional_stop_words: Optional[Collection[str]] = None, lemmatize: bool = True
) -> Filter:
    """
    Return a filter that returns the tokens that are considered stop words.

    This can be used to remove words that essentially carry no meaning.

    :param additional_stop_words: A Collection of additional stop words.
    :param lemmatize: Whether to use the lemmatized versions of the tokens.
    """
    local_stop_words = additional_stop_words if additional_stop_words else set()

    def filter_fun(doc: Collection[spacy.tokens.Token]) -> Collection[int]:
        return {
            index
            for index, token in enumerate(doc)
            if token.is_stop
            or (token.lemma_ if lemmatize else token.text) in local_stop_words
        }

    return filter_fun


def get_filter_by_vocabulary(
    vocabulary: Collection[str], lemmatize: bool = True
) -> Filter:
    """
    Return a filter that returns the tokens that are in the given vocabulary.

    This can be used to remove a specific set of unwanted words,
    e.g. too rare / frequent ones.

    :param vocabulary: The vocabulary of words to include.
    :param lemmatize: Whether to use lemmatized versions of the words.
    """

    def filter_fun(doc: Collection[spacy.tokens.Token]) -> Collection[int]:
        return {
            index
            for index, token in enumerate(doc)
            if (token.lemma_ if lemmatize else token.text) in vocabulary
        }

    return filter_fun


def get_filter_by_frequency_in_interval(
    *docs: Collection[spacy.tokens.Token],
    min_num: Optional[int] = None,
    max_num: Optional[int] = None,
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None,
    interval_open: bool = False,
    lemmatize: bool = True
) -> Filter:
    """
    Return a filter that returns the tokens with document frequency
    inside the given interval.

    Directions that are not given are considered to be unbounded.

    This can be used to remove tokens that are too rare to reason about
    or too frequent to carry much meaning.

    :param docs: The document corpus use for calculation
                 of the document frequencies.
    :param min_num: The lower bound of the interval, as an absolute number.
    :param max_num: The upper bound of the interval, as an absolute number.
    :param min_rate: The lower bound of the interval, as a relative rate.
                     Overrides min_num if given.
    :param max_rate: The upper bound of the interval, as a relative rate.
                     Overrides max_num if given.
    :param interval_open: Consider the interval to be open,
                          i.e. do not include words exactly at the boundaries.
    :param lemmatize: Whether to use the lemmatized versions of the tokens.
    """
    words_outside_interval = utils.get_words_by_df_in_interval(
        *docs,
        min_num=min_num,
        max_num=max_num,
        min_rate=min_rate,
        max_rate=max_rate,
        interval_open=interval_open,
        lemmatize=lemmatize
    )

    return get_filter_by_vocabulary(
        vocabulary=words_outside_interval, lemmatize=lemmatize
    )
