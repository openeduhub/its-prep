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
from functools import lru_cache
from collections.abc import Collection, Callable
from typing import TypeVar
import de_core_news_lg
from nlprep.types import Document, Property_Function, Split_Function, Tokens
from nlprep.utils import Spacy_defaultdict
import spacy.tokens
from pathlib import Path

# spacy NLP pipelines / models
nlp = de_core_news_lg.load()
sent_nlp = nlp.add_pipe("sentencizer")


text_to_doc_cache: Spacy_defaultdict[str] = Spacy_defaultdict(nlp)
tokens_to_doc_cache: Spacy_defaultdict[Tokens] = Spacy_defaultdict(
    lambda x: spacy.tokens.Doc(vocab=nlp.vocab, words=list(x))
)


def save_caches(directory: Path, file_prefix: str = "") -> None:
    """
    Save intermediary results into the given directory.

    In combination with load_caches, this can allow the user to skip
    unnecessary re-evaluation of already analyzed texts.
    """
    file_prefix = file_prefix + "_" if file_prefix else ""
    text_to_doc_cache.save(
        directory / f"{file_prefix}text_to_doc_cache_keys",
        directory / f"{file_prefix}text_to_doc_cache_docs",
    )
    tokens_to_doc_cache.save(
        directory / f"{file_prefix}tokens_to_doc_cache_keys",
        directory / f"{file_prefix}tokens_to_doc_cache_docs",
    )


def load_caches(directory: Path, file_prefix: str = "") -> None:
    """
    Load intermediary results from the given directory.

    In combination with save_caches, this can allow the user to skip
    unnecessary re-evaluation of already analyzed texts.
    """

    _load_text_cache(directory, file_prefix)
    _load_tokens_cache(directory, file_prefix)


def _load_text_cache(directory: Path, file_prefix: str = "") -> None:
    file_prefix = file_prefix + "_" if file_prefix else ""
    keys_path = directory / f"{file_prefix}text_to_doc_cache_keys"
    docs_path = directory / f"{file_prefix}text_to_doc_cache_docs"

    global text_to_doc_cache
    text_to_doc_cache = Spacy_defaultdict.from_file(
        default_factory=nlp,
        keys_path=keys_path,
        docs_path=docs_path,
        vocab=nlp.vocab,
    )


def _load_tokens_cache(directory: Path, file_prefix: str = "") -> None:
    file_prefix = file_prefix + "_" if file_prefix else ""
    keys_path = directory / f"{file_prefix}tokens_to_doc_cache_keys"
    docs_path = directory / f"{file_prefix}tokens_to_doc_cache_docs"

    global tokens_to_doc_cache
    default_factory = lambda x: spacy.tokens.Doc(vocab=nlp.vocab, words=list(x))
    tokens_to_doc_cache = Spacy_defaultdict.from_file(
        default_factory=default_factory,
        keys_path=keys_path,
        docs_path=docs_path,
        vocab=nlp.vocab,
    )


def _spacy_doc_from_text(text: str) -> spacy.tokens.Doc:
    return text_to_doc_cache[text]


def _spacy_doc_from_tokens(tokens: Tokens) -> spacy.tokens.Doc:
    """
    Helper function to turn tokens into processed spaCy docs,
    without re-tokenizing them
    """
    return tokens_to_doc_cache[tokens]


@lru_cache(maxsize=2**16)
def _analyze_sents(processed_doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    """Helper function to sentencize an already processed document"""
    return sent_nlp(processed_doc)


Property = TypeVar("Property")


def _raw_into_property(raw_doc: str, prop: str) -> Tokens:
    """Helper function to turn a raw document into tokens, based on property"""
    processed_doc = _spacy_doc_from_text(raw_doc)

    return Tokens(getattr(token, prop) for token in processed_doc)


def tokenize_as_words(raw_doc: str) -> Tokens:
    """Tokenize a document into its words"""
    return _raw_into_property(raw_doc, "text")


def tokenize_as_lemmas(raw_doc: str) -> Tokens:
    """Tokenize a document into its lemmas"""
    return _raw_into_property(raw_doc, "lemma_")


def _document_into_spacy_doc(doc: Document) -> spacy.tokens.Doc:
    """Transform a document into its analyzed spaCy counterpart."""

    # if the document was tokenized by spaCy, it was stored during this step
    if doc.original_text in text_to_doc_cache:
        return _spacy_doc_from_text(doc.original_text)

    # otherwise, return an analyzed version that was not tokenized again
    return _spacy_doc_from_tokens(doc.original_tokens)


def _property_from_doc(
    fun: Callable[[spacy.tokens.Doc], Collection[Property]]
) -> Property_Function[Property]:
    """
    Transform functions that act on processed spaCy documents
    to functions that act on our document representation.

    This allows us to easily define processing steps within spaCy's context,
    without having to deal with mapping its internal document representation
    to ours.

    We store a mapping between the original text and the processed document,
    such that each sub-document does not need to be processed again.
    """

    def wrapped_fun(doc: Document) -> Collection[Property]:
        processed_doc = _document_into_spacy_doc(doc)
        return fun(processed_doc)

    return wrapped_fun


def _sentencizer_from_doc(
    fun: Callable[[spacy.tokens.Doc], Collection[Collection[Property]]]
) -> Split_Function[Property]:
    """
    Analogous to the decorator for property functions, but for sentencizers.

    Because the default processing pipeline does not necessarily include
    a sentencizer, run the processed document through one
    before passing it onto the given function.
    """

    def wrapped_fun(doc: Document) -> Collection[Collection[Property]]:
        processed_doc = _document_into_spacy_doc(doc)
        return fun(_analyze_sents(processed_doc))

    return wrapped_fun


@_property_from_doc
def get_upos(processed_doc: spacy.tokens.Doc) -> Collection[str]:
    """The universal POS tags of each token"""
    return [token.pos_ for token in processed_doc]


@_property_from_doc
def is_stop(processed_doc: spacy.tokens.Doc) -> Collection[bool]:
    """Indicators whether each token is a stop word"""
    return [token.is_stop for token in processed_doc]


@_property_from_doc
def lemmatize(processed_doc: spacy.tokens.Doc) -> Collection[str]:
    """The lemmatized version of each token"""
    return [token.lemma_ for token in processed_doc]


@_sentencizer_from_doc
def into_sentences(processed_doc: spacy.tokens.Doc) -> Collection[Collection[str]]:
    """Split the document by its sentences"""
    return [[token.text for token in sent] for sent in processed_doc.sents]


@_sentencizer_from_doc
def into_sentences_lemmatized(
    processed_doc: spacy.tokens.Doc,
) -> Collection[Collection[str]]:
    """Split the document by its sentences, with lemmatization"""
    return [[token.lemma_ for token in sent] for sent in processed_doc.sents]
