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
from wloprep.types import Document
import spacy.tokens
import de_dep_news_trf

nlp = de_dep_news_trf.load()
DOCUMENT_CACHE: dict[tuple[str], spacy.tokens.Doc] = dict()

T = TypeVar("T")


def from_doc(fun: Callable[[spacy.tokens.Doc], T]) -> Callable[[Document], T]:
    """
    Transform functions that act on processed spaCy documents
    to functions that act on generic, tokenized documents.
    """

    def wrapped_fun(doc: Document) -> T:
        processed_doc = DOCUMENT_CACHE.setdefault(doc, nlp(" ".join(doc)))
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
