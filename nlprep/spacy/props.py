"""
A collection of functions that compute various text properties,
based on the spaCy library and its de_dep_news_trf model in particular.

See https://huggingface.co/spacy/de_dep_news_trf for more details on the model.

To reduce decrease unnecessary calculations,
cache spaCy representations of already processed documents.

Note that even though these functions are defined on
spaCy-specific document representations,
they will actually act on the internal Document representation.
"""
import nlprep.spacy.utils as utils
from nlprep.types import Tokens

import spacy.tokens


def tokenize_as_words(raw_doc: str) -> Tokens:
    """Tokenize a document into its words"""
    return utils.raw_into_property(raw_doc, "text")


def tokenize_as_lemmas(raw_doc: str) -> Tokens:
    """Tokenize a document into its lemmas"""
    return utils.raw_into_property(raw_doc, "lemma_")


@utils.property_from_doc
def get_upos(processed_doc: spacy.tokens.Doc) -> list[str]:
    """The universal POS tags of each token"""
    return [token.pos_ for token in processed_doc]


@utils.property_from_doc
def is_stop(processed_doc: spacy.tokens.Doc) -> list[bool]:
    """Indicators whether each token is a stop word"""
    return [token.is_stop for token in processed_doc]


@utils.property_from_doc
def lemmatize(processed_doc: spacy.tokens.Doc) -> list[str]:
    """The lemmatized version of each token"""
    return [token.lemma_ for token in processed_doc]


@utils.sentencizer_from_doc
def into_sentences(processed_doc: spacy.tokens.Doc) -> list[list[str]]:
    """Split the document by its sentences"""
    return [[token.text for token in sent] for sent in processed_doc.sents]


@utils.sentencizer_from_doc
def into_sentences_lemmatized(
    processed_doc: spacy.tokens.Doc,
) -> list[list[str]]:
    """Split the document by its sentences, with lemmatization"""
    return [[token.lemma_ for token in sent] for sent in processed_doc.sents]
