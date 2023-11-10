from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import de_core_news_lg
from nlprep.types import Document, Property, Property_Function, Split_Function, Tokens
from nlprep.utils import Spacy_defaultdict

import spacy.tokens

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


def load_caches(directory: Path, file_prefix: str = "") -> None:
    """
    Load intermediary results from the given directory.

    In combination with save_caches, this can allow the user to skip
    unnecessary re-evaluation of already analyzed texts.
    """

    _load_text_cache(directory, file_prefix)
    _load_tokens_cache(directory, file_prefix)


def spacy_doc_from_text(text: str) -> spacy.tokens.Doc:
    return text_to_doc_cache[text]


def spacy_doc_from_tokens(tokens: Tokens) -> spacy.tokens.Doc:
    """
    Helper function to turn tokens into processed spaCy docs,
    without re-tokenizing them
    """
    return tokens_to_doc_cache[tokens]


def raw_into_property(raw_doc: str, prop: str) -> Tokens:
    """Helper function to turn a raw document into tokens, based on property"""
    processed_doc = spacy_doc_from_text(raw_doc)

    return Tokens(getattr(token, prop) for token in processed_doc)


def document_into_spacy_doc(doc: Document) -> spacy.tokens.Doc:
    """Transform a document into its analyzed spaCy counterpart."""

    # if the document was tokenized by spaCy, it was stored during this step
    if doc.original_text in text_to_doc_cache:
        return spacy_doc_from_text(doc.original_text)

    # otherwise, return an analyzed version that was not tokenized again
    return spacy_doc_from_tokens(doc.original_tokens)


def property_from_doc(
    fun: Callable[[spacy.tokens.Doc], Sequence[Property]]
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

    def wrapped_fun(doc: Document) -> Sequence[Property]:
        processed_doc = document_into_spacy_doc(doc)
        return fun(processed_doc)

    return wrapped_fun


@lru_cache(maxsize=2**16)
def _analyze_sents(processed_doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    """Helper function to sentencize an already processed document"""
    return sent_nlp(processed_doc)


def sentencizer_from_doc(
    fun: Callable[[spacy.tokens.Doc], Sequence[Sequence[Property]]]
) -> Split_Function[Property]:
    """
    Analogous to the decorator for property functions, but for sentencizers.

    Because the default processing pipeline does not necessarily include
    a sentencizer, run the processed document through one
    before passing it onto the given function.
    """

    def wrapped_fun(doc: Document) -> Sequence[Sequence[Property]]:
        processed_doc = document_into_spacy_doc(doc)
        return fun(_analyze_sents(processed_doc))

    return wrapped_fun
