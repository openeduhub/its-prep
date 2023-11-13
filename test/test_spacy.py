from collections.abc import Callable, Collection
import functools
from typing import Text
from nlprep.core import tokenize_documents
from nlprep.types import Document, Property_Function, Split_Function, Tokens
import test.strategies as nlp_st
from pathlib import Path

import nlprep.spacy.props as nlp
from hypothesis import given
from hypothesis import strategies as st


@given(
    nlp_st.texts,
    st.sampled_from([nlp.tokenize_as_lemmas, nlp.tokenize_as_words]),
)
def test_tokenizers_do_something(text: str, tokenizer: Callable[[str], Tokens]):
    result = tokenizer(text)

    if text:
        assert result


@given(
    nlp_st.documents,
    st.sampled_from([nlp.get_upos, nlp.lemmatize, nlp.is_stop]),
)
def test_properties_have_correct_len(doc: Document, fun: Property_Function):
    result = fun(doc)
    assert len(result) == len(doc.original_tokens)


@given(
    nlp_st.documents,
    st.sampled_from([nlp.into_sentences, nlp.into_sentences_lemmatized]),
)
def test_splitters_have_correct_len(doc: Document, splitter: Split_Function):
    split_result = splitter(doc)
    assert sum(len(result) for result in split_result) == len(doc.original_tokens)


@given(nlp_st.texts)
def test_text_cache(text: str):
    doc = nlp.tokenize_as_words(text)

    assert text in nlp.utils.text_to_doc_cache
    for token_doc, token_cache in zip(doc, nlp.utils.text_to_doc_cache[text]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.tokens)
def test_tokens_cache(tokens: Tokens):
    doc = nlp.utils.spacy_doc_from_tokens(tokens)

    assert tokens in nlp.utils.tokens_to_doc_cache
    for token_doc, token_cache in zip(doc, nlp.utils.tokens_to_doc_cache[tokens]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.texts)
def test_text_cache_storage(text: str):
    doc = nlp.tokenize_as_words(text)

    path = Path("/tmp/nlprep-test")
    path.mkdir(parents=True, exist_ok=True)

    # store the cache
    nlp.utils.save_caches(path, file_prefix="pytest")
    # delete the text from the cache
    del nlp.utils.text_to_doc_cache[text]
    # load the cache
    nlp.utils.load_caches(path, file_prefix="pytest")
    # delete the cache files
    [file.unlink() for file in path.glob("pytest*")]

    assert text in nlp.utils.text_to_doc_cache
    for token_doc, token_cache in zip(doc, nlp.utils.text_to_doc_cache[text]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.tokens)
def test_tokens_cache_storage(tokens: Tokens):
    doc = nlp.utils.spacy_doc_from_tokens(tokens)

    # store the cache
    nlp.utils.save_caches(Path("/tmp"), file_prefix="pytest")
    # delete the tokens from the cache
    del nlp.utils.tokens_to_doc_cache[tokens]
    # load the cache
    nlp.utils.load_caches(Path("/tmp"), file_prefix="pytest")

    assert tokens in nlp.utils.tokens_to_doc_cache
    for token_doc, token_cache in zip(doc, nlp.utils.tokens_to_doc_cache[tokens]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.texts)
def test_noun_chunks(text: str):
    # the tested function only works for documents tokenized by spacy directly
    doc = list(tokenize_documents([text], nlp.tokenize_as_words))[0]
    chunks = nlp.noun_chunks(doc)

    # property functions must have the same length as the original document
    assert len(chunks) == len(doc)


def test_noun_chunks_without_nouns():
    text = "gehen, laufen, schwimmen"
    doc = list(tokenize_documents([text], nlp.tokenize_as_words))[0]
    chunks = nlp.noun_chunks(doc)

    assert all(chunk is None for chunk in chunks)


def test_noun_chunks_with_nouns():
    text = "Ein hungriger Hund geht in einem schönen See baden."
    doc = list(tokenize_documents([text], nlp.tokenize_as_words))[0]
    chunks = nlp.noun_chunks(doc)

    hund_chunks = chunks[:3]
    see_chunks = chunks[5:8]
    none_chunks = list(chunks[3:5]) + [chunks[-1]]

    # assert that the correct tokens are assigned Nones / ints
    assert all(chunk is not None for chunk in hund_chunks)
    assert all(chunk is not None for chunk in see_chunks)
    assert all(chunk is None for chunk in none_chunks)

    # assert that all chunk IDs are correct
    assert all(
        chunk_a == chunk_b for chunk_a, chunk_b in zip(hund_chunks, hund_chunks[1:])
    )
    assert all(
        chunk_a == chunk_b for chunk_a, chunk_b in zip(see_chunks, see_chunks[1:])
    )
