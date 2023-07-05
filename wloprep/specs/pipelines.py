"""
A collection of pipelines that were used for various tasks.
Here, 'pipeline' refers to a sequence of filtering functions.

These can be used as-is, or as guidance for defining new pipelines.
To apply these pipelines to a document corpus, use the
filter.apply_filters function.
"""
from collections.abc import Collection, Callable
from typing import Any
from wloprep.types import Document, Filter
from wloprep.core import negated
import wloprep.specs.filters as filters
import wloprep.specs.collections as cols
import wloprep.spacy.props as spacy_props


def get_pipeline_generic_topic_modeling(
    *docs: Document,
    get_upos_fun: Callable[[Document], Collection[str]],
    is_stop_fun: Callable[[Document], Collection[bool]],
    lemmatize_fun: Callable[[Document], Collection[str]],
    ignored_upos_tags: Collection[str],
    ignored_lemmas: Collection[str],
    interval_spec: dict[str, Any],
) -> Collection[Filter]:
    """
    Pipeline of filter functions used during pre-processing for topic modeling.

    1. Filter based on unwanted universal POS tags.
    2. Filter out stop words.
    3. Filter out unwanted lemmas.
    4. Filter out very rare and very frequent words.

    :param interval_spec: Specification of the document frequency interval
                          that tokens must fall into. See documentation of
                          filter_specs.get_filter_by_frequency_in_interval
    """
    return [
        # filter by upos tags
        negated(filters.get_filter_by_property(get_upos_fun, ignored_upos_tags)),
        # filter by stop words
        negated(filters.get_filter_by_boolean_fun(is_stop_fun)),
        # filter by ignored lemmas
        negated(filters.get_filter_by_property(lemmatize_fun, ignored_lemmas)),
        # filter by document frequency of lemmatized tokens
        filters.get_filter_by_frequency(docs, lemmatize_fun, **interval_spec),
    ]


def get_pipeline_poc_topic_modeling(*docs: Document):
    """The particular pipeline used for the PoC topic modeling application."""
    return get_pipeline_generic_topic_modeling(
        *docs,
        lemmatize_fun=spacy_props.lemmatize,
        get_upos_fun=spacy_props.get_upos,
        is_stop_fun=spacy_props.is_stop,
        # ignore punctuation and white-space
        ignored_upos_tags={"PUNCT", "SPACE"},
        # lemmas must be in at least five and at most 25% of documents
        interval_spec={
            "min_num": 5,
            "max_rate": 0.25,
            "open_interval": False,
        },
        # ignore the following lemmas
        ignored_lemmas=set().union(
            cols.symbols,
            cols.fillers,
            cols.lrts,
            cols.regions,
            cols.sources,
            cols.target_groups,
        ),
    )
