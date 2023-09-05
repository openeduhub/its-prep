"""
A collection of pipelines that were used for various tasks.
Here, 'pipeline' refers to a sequence of filtering functions.

These can be used as-is, or as guidance for defining new pipelines.
To apply these pipelines to a document corpus, use the
filter.apply_filters function.
"""
from collections.abc import Collection, Callable
from typing import Any, TypeVar
from nlprep.core import apply_filters
from nlprep.types import Document, Pipeline, Property_Function
import nlprep.specs.filters as filters
import nlprep.specs.collections as cols
import nlprep.spacy.props as nlp

Upos = TypeVar("Upos")
Lemma = TypeVar("Lemma")


def apply_generic_topic_modeling(
    docs: Collection[Document],
    get_upos_fun: Property_Function[Upos],
    is_stop_fun: Property_Function[bool],
    lemmatize_fun: Property_Function[Lemma],
    ignored_upos_tags: Collection[Upos],
    ignored_lemmas: Collection[Lemma],
    required_df_interval: dict[str, Any],
) -> Collection[Document]:
    """
    Pipeline of filter functions used during pre-processing for topic modeling.

    1. Filter based on unwanted universal POS tags.
    2. Filter out stop words.
    3. Filter out unwanted lemmas.
    4. Filter out very rare and very frequent words.

    :param required_df_interval:
      Specification of the document frequency interval
      that tokens must fall into.
      See documentation of filter_specs.get_filter_by_frequency_in_interval.
    """
    # filter by everything but document frequency
    initial_pipeline = [
        # filter by upos tags
        filters.negated(
            filters.get_filter_by_property(get_upos_fun, ignored_upos_tags)
        ),
        # filter by stop words
        filters.negated(filters.get_filter_by_bool_fun(is_stop_fun)),
        # filter by ignored lemmas
        filters.negated(filters.get_filter_by_property(lemmatize_fun, ignored_lemmas)),
    ]

    docs = list(apply_filters(docs, initial_pipeline))

    # filter by document frequency of lemmatized tokens
    frequency_pipeline = [
        filters.get_filter_by_frequency(
            docs,
            lemmatize_fun,
            **required_df_interval,
        )
    ]

    return list(apply_filters(docs, frequency_pipeline))


def apply_poc_topic_modeling(
    docs: Collection[Document],
    *,
    required_df_interval: dict[str, Any] = {
        "min_num": 5,
        "max_rate": 0.25,
        "interval_open": False,
        "count_only_selected": True,
    },
    ignored_upos_tags: Collection[str] = {"PUNCT", "SPACE"},
    ignored_lemmas=set().union(
        cols.symbols,
        cols.fillers,
        cols.lrts,
        cols.sources,
        cols.target_audiences,
    ),
) -> Collection[Document]:
    """The particular pipeline used for the PoC topic modeling application."""
    return apply_generic_topic_modeling(
        docs,
        lemmatize_fun=nlp.lemmatize,
        get_upos_fun=nlp.get_upos,
        is_stop_fun=nlp.is_stop,
        # ignore punctuation and white-space
        ignored_upos_tags=ignored_upos_tags,
        # lemmas must be in at least five and at most 25% of documents
        required_df_interval=required_df_interval,
        # ignore the following lemmas
        ignored_lemmas=ignored_lemmas,
    )
