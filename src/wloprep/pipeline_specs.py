"""
A collection of pipelines that were used for various tasks.
Here, 'pipeline' refers to a sequence of filtering functions.

These can be used as-is, or as guidance for defining new pipelines.
To apply these pipelines to a document corpus, use the
filter.apply_filters function.
"""
from collections.abc import Collection
from typing import Any, Optional
import wloprep.filter as filter
import wloprep.filter_specs as filters
from spacy.tokens import Doc


def get_pipeline_generic_topic_modeling(
    ignored_upos_tags: Optional[Collection[str]] = None,
    ignored_lemmas: Optional[Collection[str]] = None,
    docs: Optional[Collection[Doc]] = None,
    interval_spec: Optional[dict[str, Any]] = None,
) -> Collection[filter.Filter]:
    """
    Pipeline of filter functions used during pre-processing for topic modeling.

    1. Filter based on unwanted universal POS tags.
    2. Filter out stop words.
    3. Filter out unwanted lemmas.
    4. Filter out very rare and very frequent words.

    :param ignored_upos_tags: Universal POS tags to remove.
    :param ignored_lemmas: Lemmas to remove.
    :param docs: The document corpus to use to calculate document frequencies.
                 Used when filtering out rare and frequent words.
    :param interval_spec: Specification of the document frequency interval
                          that tokens must fall into. See documentation of
                          filter_specs.get_filter_by_frequency_in_interval
    """
    identity = lambda doc: set(range(len(doc)))
    return [
        filter.exclude(filters.get_filter_by_upos(upos_tags=ignored_upos_tags))
        if ignored_upos_tags
        else identity,
        filter.exclude(filters.get_filter_by_stop_words()),
        filter.exclude(
            filters.get_filter_by_vocabulary(vocabulary=ignored_lemmas, lemmatize=True)
        )
        if ignored_lemmas
        else identity,
        filters.get_filter_by_frequency(*docs, **interval_spec)
        if docs and interval_spec
        else identity,
    ]


def get_pipeline_poc_topic_modeling(*docs: Doc):
    """The particular pipeline used for the PoC topic modeling application."""
    return get_pipeline_generic_topic_modeling(
        docs=docs,
        # ignore punctuation and white-space
        ignored_upos_tags={"PUNCT", "SPACE"},
        # tokens must be in at least five and at most 25% of documents
        interval_spec={
            "min_num": 5,
            "max_rate": 0.25,
            "open_interval": False,
            "lemmatize": True,
        },
        # ignore the following lemmas
        ignored_lemmas={
            "-",
            "--",
            "&",
            "|",
            "Thema",
            "Video",
            "interaktiv",
            "Überblick",
            "de",
            "Internetseite",
            "eBildungslabor",
            "Lesepfad",
            "Lexikon",
            "kostenlos",
            "Arbeitsblatt",
            "BR",
            "Podcast",
            "Audiobeitrag",
            "Universität",
            "Projekt",
            "Kompetenz",
            "Schülerin",
            "Schüler",
            "Schüler*in",
            "Lernpfad",
            "spannend",
            "zeigen",
            "um",
            "Methode",
            "Lernort",
            "umfassend",
            "Anleitung",
            "thematisieren",
            "geben",
            "Schritt",
            "downloaden",
            "dein",
            "innen",
            "Veranschaulichung",
            "Baden-Württemberg",
            "Unterricht",
            "behandeln",
            "ARD-Faktenfinder",
            "Verifikationsteam",
            "ARD",
            "Klexikon",
            "frei",
            "Internet-Lexikon",
            "Wiki",
            "kindgerecht",
            "Inhalt",
            "Ansprache",
            "Altersgruppe",
            "Online-Spiel",
            "anhand",
            "erfahren",
            "Schüler*innen",
            "eigentlich",
            "einfach",
            "anwenden",
            "Spiel",
            "Lerneffekt",
            "PHSZ",
            "Unterrichtseinheit",
            "zahlreich",
            "Material",
            "thematisieren",
            "spielerisch",
            "einsteigen",
            "Landesbildungsserver",
            "Unterrichtsbaustein",
            "weiterführend",
            "Link",
            "Volksschule",
            "Sammlung",
            "Beispiel",
            "YouTube",
            "Youtube",
            "Schüler/-innen",
        },
    )
