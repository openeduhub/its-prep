from collections.abc import Callable, Collection
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

    assert text in nlp.text_to_doc_cache
    for token_doc, token_cache in zip(doc, nlp.text_to_doc_cache[text]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.tokens)
def test_tokens_cache(tokens: Tokens):
    doc = nlp._spacy_doc_from_tokens(tokens)

    assert tokens in nlp.tokens_to_doc_cache
    for token_doc, token_cache in zip(doc, nlp.tokens_to_doc_cache[tokens]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.texts)
def test_text_cache_storage(text: str):
    doc = nlp.tokenize_as_words(text)

    # store the cache
    nlp.save_caches(Path("/tmp"), file_prefix="pytest")
    # delete the text from the cache
    del nlp.text_to_doc_cache[text]
    # load the cache
    nlp.load_caches(Path("/tmp"), file_prefix="pytest")

    assert text in nlp.text_to_doc_cache
    for token_doc, token_cache in zip(doc, nlp.text_to_doc_cache[text]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.tokens)
def test_tokens_cache_storage(tokens: Tokens):
    doc = nlp._spacy_doc_from_tokens(tokens)

    # store the cache
    nlp.save_caches(Path("/tmp"), file_prefix="pytest")
    # delete the tokens from the cache
    del nlp.tokens_to_doc_cache[tokens]
    # load the cache
    nlp.load_caches(Path("/tmp"), file_prefix="pytest")

    assert tokens in nlp.tokens_to_doc_cache
    for token_doc, token_cache in zip(doc, nlp.tokens_to_doc_cache[tokens]):
        assert str(token_doc) == str(token_cache)
