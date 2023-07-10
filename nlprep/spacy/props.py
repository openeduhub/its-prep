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
import spacy.tokens
import de_dep_news_trf

nlp = de_dep_news_trf.load()
DOCUMENT_CACHE: dict[str, spacy.tokens.Doc] = dict()

T = TypeVar("T")


def _raw_into_property(raw_doc: str, prop: str) -> tuple[str, ...]:
    processed_doc = nlp(raw_doc)

    # cache the processed raw text for later use
    DOCUMENT_CACHE[raw_doc] = processed_doc

    return tuple(getattr(token, prop) for token in processed_doc)


def raw_into_words(raw_doc: str) -> tuple[str, ...]:
    return _raw_into_property(raw_doc, "text")


def raw_into_lemmas(raw_doc: str) -> tuple[str, ...]:
    return _raw_into_property(raw_doc, "lemma_")


def from_doc(fun: Callable[[spacy.tokens.Doc], T]) -> Callable[[Document], T]:
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
        processed_doc = DOCUMENT_CACHE.setdefault(
            doc.original_text, nlp(doc.original_text)
        )
        return fun(processed_doc)

    return wrapped_fun


@from_doc
def get_upos(processed_doc: spacy.tokens.Doc) -> Collection[str]:
    return [token.pos_ for token in processed_doc]


@from_doc
def is_stop(processed_doc: spacy.tokens.Doc) -> Collection[bool]:
    return [token.is_stop for token in processed_doc]


@from_doc
def lemmatize(processed_doc: spacy.tokens.Doc) -> Collection[str]:
    return [token.lemma_ for token in processed_doc]


@from_doc
def into_sentences(processed_doc: spacy.tokens.Doc) -> Collection[Collection[str]]:
    return [[token.text for token in sent] for sent in processed_doc.sents]


@from_doc
def into_sentences_lemmatized(
    processed_doc: spacy.tokens.Doc,
) -> Collection[Collection[str]]:
    return [[token.lemma_ for token in sent] for sent in processed_doc.sents]
