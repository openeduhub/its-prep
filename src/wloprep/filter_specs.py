"""
A collection of filter functions used in the pipelines from pipeline_spec.

Because these filter functions are not completely static,
i.e. they should be parameterized first or depend on the input data,
they are not defined directly, but created factory functions instead.

These can be used as-is inside of pipeline definitions,
or as guidance for defining further filtering functions.
"""

from collections.abc import Callable, Collection
from typing import Optional

from wloprep.filter import Filter, Document


def get_filter_by_upos(
    get_upos_fun: Callable[[Document], Collection[str]],
    upos_tags: Collection[str],
) -> Filter:
    """
    Return a filter that returns the tokens depending on their UPOS tag.

    This can be used to remove words that tend to be relevant to the task,
    such as prepositions.

    :param get_upos: The function to use to analyze the document,
                     obtaining each token's universal POS tag.
    :param upos_tags: The collection of universal POS tags to look for.
    """

    def filter_fun(doc: Document) -> Collection[int]:
        return {
            index for index, upos in enumerate(get_upos_fun(doc)) if upos in upos_tags
        }

    return filter_fun


def get_filter_by_stop_words(
    is_stop_fun: Callable[[Document], Collection[bool]]
) -> Filter:
    """
    Return a filter that returns the tokens that are considered stop words.

    This can be used to remove words that essentially carry no meaning.

    :param analyze_fun: The function to use to analyze the document.
    """

    def filter_fun(doc: Document) -> Collection[int]:
        return {index for index, is_stop in enumerate(is_stop_fun(doc)) if is_stop}

    return filter_fun


def get_filter_by_vocabulary(
    lemmatize_fun: Callable[[Document], Collection[str]], vocabulary: Collection[str]
) -> Filter:
    """
    Return a filter that returns the tokens that are in the given vocabulary.

    This can be used to remove a specific set of unwanted words,
    e.g. from a fixed vocabulary of unwanted words.

    :param lemmatize_fun: The function to use to lemmatize the document.
    :param vocabulary: The vocabulary of words to include.
    """

    def filter_fun(doc: Document) -> Collection[int]:
        return {
            index
            for index, lemma in enumerate(lemmatize_fun(doc))
            if lemma in vocabulary
        }

    return filter_fun


def get_words_by_df_in_interval(
    docs: Collection[Document],
    lemmatize_fun: Callable[[Document], Collection[str]],
    min_num: Optional[int] = None,
    max_num: Optional[int] = None,
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None,
    interval_open: bool = False,
) -> Collection[str]:
    """
    Return the words with document frequency outside the given interval.

    Directions that are not given are considered to be unbounded.

    This can be used to remove tokens that are too rare to reason about
    or too frequent to carry much meaning.

    :param docs: The document corpus to process.
    :param lemmatize_fun: The function to use to lemmatize the document.
    :param min_num: The lower bound of the interval, as the absolute number.
    :param max_num: The upper bound of the interval, as the absolute number.
    :param min_rate: The lower bound of the interval, as the relative rate.
    :param max_rate: The upper bound of the interval, as the relative rate.
    :param interval_open: Consider the interval to be open,
                          i.e. do not include words exactly at the boundaries.
    """
    # override the interval boundaries according to the given rates
    if min_rate:
        min_num = int(len(docs) * min_rate)

    if max_rate:
        max_num = int(len(docs) * max_rate)

    # helper functions to determine whether a count is in the interval
    def right_of_min(count: int) -> bool:
        if min_num is None:
            return True

        if interval_open:
            return count < min_num
        else:
            return count <= min_num

    def left_of_max(count: int) -> bool:
        if max_num is None:
            return True

        if interval_open:
            return count > max_num
        else:
            return count >= max_num

    def in_interval(count: int) -> bool:
        return right_of_min(count) and left_of_max(count)

    # helper function to compute document frequencies
    def get_document_freqs() -> dict[str, int]:
        docs_as_sets = [{lemma for lemma in lemmatize_fun(doc)} for doc in docs]
        vocabulary = set().union(*docs_as_sets)
        return {
            lemma: sum([lemma in document for document in docs_as_sets])
            for lemma in vocabulary
        }

    dfs = get_document_freqs()
    return {lemma for lemma, count in dfs.items() if in_interval(count)}


def get_filter_by_frequency(
    docs: Collection[Document],
    lemmatize_fun: Callable[[Document], Collection[str]],
    min_num: Optional[int] = None,
    max_num: Optional[int] = None,
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None,
    interval_open: bool = False,
) -> Filter:
    """
    Return a filter that returns the tokens with document frequency
    inside the given interval.

    Directions that are not given are considered to be unbounded.

    This can be used to remove tokens that are too rare to reason about
    or too frequent to carry much meaning.

    :param docs: The document corpus use for calculation
                 of the document frequencies.
    :param lemmatize_fun: The function to use to lemmatize the document.
    :param min_num: The lower bound of the interval, as an absolute number.
    :param max_num: The upper bound of the interval, as an absolute number.
    :param min_rate: The lower bound of the interval, as a relative rate.
                     Overrides min_num if given.
    :param max_rate: The upper bound of the interval, as a relative rate.
                     Overrides max_num if given.
    :param interval_open: Consider the interval to be open,
                          i.e. do not include words exactly at the boundaries.
    """
    words_inside_interval = get_words_by_df_in_interval(
        lemmatize_fun=lemmatize_fun,
        docs=docs,
        min_num=min_num,
        max_num=max_num,
        min_rate=min_rate,
        max_rate=max_rate,
        interval_open=interval_open,
    )

    return get_filter_by_vocabulary(
        lemmatize_fun=lemmatize_fun, vocabulary=words_inside_interval
    )


def get_filter_by_sent_len(
    into_sentences_fun: Callable[[Document], Collection[Collection]],
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    interval_open: bool = False,
) -> Filter:
    # helper functions to determine whether a length is in the interval
    def right_of_min(count: int) -> bool:
        if min_len is None:
            return True

        if interval_open:
            return count < min_len
        else:
            return count <= min_len

    def left_of_max(count: int) -> bool:
        if max_len is None:
            return True

        if interval_open:
            return count > max_len
        else:
            return count >= max_len

    def in_interval(length: int) -> bool:
        return right_of_min(length) and left_of_max(length)

    def filter_fun(doc: Document) -> Collection[int]:
        len_by_word = [len(sent) for sent in into_sentences_fun(doc) for _ in sent]
        return {
            index for index, length in enumerate(len_by_word) if in_interval(length)
        }

    return filter_fun
