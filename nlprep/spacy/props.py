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
from collections.abc import Collection, Callable
from typing import TypeVar
from nlprep.types import Document
from nlprep.utils import defaultdict_keyed
import spacy.tokens
import de_dep_news_trf

nlp = de_dep_news_trf.load()
DOCUMENT_CACHE: defaultdict_keyed[str, spacy.tokens.Doc] = defaultdict_keyed(nlp)

T = TypeVar("T")


def _raw_into_property(raw_doc: str, prop: str) -> tuple[str, ...]:
    processed_doc = DOCUMENT_CACHE[raw_doc]

    return tuple(getattr(token, prop) for token in processed_doc)


def raw_into_words(raw_doc: str) -> tuple[str, ...]:
    """Tokenize a document into its words"""
    return _raw_into_property(raw_doc, "text")


def raw_into_lemmas(raw_doc: str) -> tuple[str, ...]:
    """Tokenize a document into its lemmas"""
    return _raw_into_property(raw_doc, "lemma_")


def _from_doc(fun: Callable[[spacy.tokens.Doc], T]) -> Callable[[Document], T]:
    """
    Transform functions that act on processed spaCy documents
    to functions that act on our document representation.

    This allows us to easily define processing steps within spaCy's context,
    without having to deal with mapping its internal document representation
    to ours.

    We store a mapping between the original text and the processed document,
    such that each sub-document does not need to be processed again.
    """

    def wrapped_fun(doc: Document) -> T:
        return fun(DOCUMENT_CACHE[doc.original_text])

    return wrapped_fun


@_from_doc
def get_upos(processed_doc: spacy.tokens.Doc) -> Collection[str]:
    """The universal POS tags of each token"""
    return [token.pos_ for token in processed_doc]


@_from_doc
def is_stop(processed_doc: spacy.tokens.Doc) -> Collection[bool]:
    """Indicators whether each token is a stop word"""
    return [token.is_stop for token in processed_doc]


@_from_doc
def lemmatize(processed_doc: spacy.tokens.Doc) -> Collection[str]:
    """The lemmatized version of each token"""
    return [token.lemma_ for token in processed_doc]


@_from_doc
def into_sentences(processed_doc: spacy.tokens.Doc) -> Collection[Collection[str]]:
    """Split the document by its sentences"""
    return [[token.text for token in sent] for sent in processed_doc.sents]


@_from_doc
def into_sentences_lemmatized(
    processed_doc: spacy.tokens.Doc,
) -> Collection[Collection[str]]:
    """Split the document by its sentences, with lemmatization"""
    return [[token.lemma_ for token in sent] for sent in processed_doc.sents]
