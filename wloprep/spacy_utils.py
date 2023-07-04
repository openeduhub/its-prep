from collections.abc import Collection
from wloprep.types import Document
import spacy.tokens
import de_dep_news_trf

nlp = de_dep_news_trf.load()
DOCUMENT_CACHE: dict[tuple[str], spacy.tokens.Doc] = dict()


def get_upos_fun(doc: Document) -> Collection[str]:
    processed_doc = DOCUMENT_CACHE.setdefault(doc, nlp(" ".join(doc)))
    return [token.pos_ for token in processed_doc]


def is_stop_fun(doc: Document) -> Collection[bool]:
    processed_doc = DOCUMENT_CACHE.setdefault(doc, nlp(" ".join(doc)))
    return [token.is_stop for token in processed_doc]


def lemmatize_fun(doc: Document) -> Collection[str]:
    processed_doc = DOCUMENT_CACHE.setdefault(doc, nlp(" ".join(doc)))
    return [token.lemma_ for token in processed_doc]


def into_sentences_fun(doc: Document) -> Collection[Collection[str]]:
    processed_doc = DOCUMENT_CACHE.setdefault(doc, nlp(" ".join(doc)))
    return [[token.text for token in sent] for sent in processed_doc.sents]


def into_sentences_fun_lemmatized(doc: Document) -> Collection[Collection[str]]:
    processed_doc = DOCUMENT_CACHE.setdefault(doc, nlp(" ".join(doc)))
    return [[token.lemma_ for token in sent] for sent in processed_doc.sents]
