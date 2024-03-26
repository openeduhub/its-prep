import test.strategies as nlp_st
from collections.abc import Callable
from pathlib import Path

import its_prep.spacy.props as nlp
from hypothesis import given, settings
from hypothesis import strategies as st
from its_prep.core import tokenize_documents
from its_prep.types import Document, Property_Function, Split_Function, Tokens


@given(
    nlp_st.texts,
    st.sampled_from([nlp.tokenize_as_lemmas, nlp.tokenize_as_words]),
)
@settings(deadline=None)
def test_tokenizers_do_something(text: str, tokenizer: Callable[[str], Tokens]):
    result = tokenizer(text)

    if text:
        assert result


@given(
    nlp_st.documents,
    st.sampled_from([nlp.get_upos, nlp.lemmatize, nlp.is_stop]),
)
@settings(deadline=None)
def test_properties_have_correct_len(doc: Document, fun: Property_Function):
    result = fun(doc)
    assert len(result) == len(doc.original_tokens)


@given(
    nlp_st.documents,
    st.sampled_from([nlp.into_sentences, nlp.into_sentences_lemmatized]),
)
@settings(deadline=None)
def test_splitters_have_correct_len(doc: Document, splitter: Split_Function):
    split_result = splitter(doc)
    assert sum(len(result) for result in split_result) == len(doc.original_tokens)


@given(nlp_st.texts)
@settings(deadline=None)
def test_text_cache(text: str):
    doc = nlp.tokenize_as_words(text)

    for token_doc, token_cache in zip(
        doc, nlp.utils.original_spacy_doc_from_text(text)
    ):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.tokens)
@settings(deadline=None)
def test_tokens_cache(tokens: Tokens):
    doc = nlp.utils.spacy_doc_from_tokens(tokens)

    assert tokens in nlp.utils._tokens_cache
    for token_doc, token_cache in zip(doc, nlp.utils._tokens_cache[tokens]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.texts)
@settings(deadline=None)
def test_text_cache_storage(text: str):
    doc = nlp.tokenize_as_words(text)

    path = Path("/tmp/its-prep-test")
    path.mkdir(parents=True, exist_ok=True)

    # store the cache
    nlp.utils.save_caches(path, file_prefix="pytest")
    # delete the text from the cache
    del nlp.utils._text_cache_original[text]
    # load the cache
    nlp.utils.load_caches(path, file_prefix="pytest")
    # delete the cache files
    [file.unlink() for file in path.glob("pytest*")]

    assert text in nlp.utils._text_cache_original
    for token_doc, token_cache in zip(doc, nlp.utils._text_cache_original[text]):
        assert str(token_doc) == str(token_cache)


@given(nlp_st.tokens)
@settings(deadline=None)
def test_tokens_cache_storage(tokens: Tokens):
    doc = nlp.utils.spacy_doc_from_tokens(tokens)

    # store the cache
    nlp.utils.save_caches(Path("/tmp"), file_prefix="pytest")
    # delete the tokens from the cache
    del nlp.utils._tokens_cache[tokens]
    # load the cache
    nlp.utils.load_caches(Path("/tmp"), file_prefix="pytest")

    assert tokens in nlp.utils._tokens_cache
    for token_doc, token_cache in zip(doc, nlp.utils._tokens_cache[tokens]):
        assert str(token_doc) == str(token_cache)


@given(st.booleans(), st.booleans())
@settings(deadline=None)
def test_tokenize_merges(merge_named_entities: bool, merge_noun_chunks: bool):
    text = "Deutschland ist ein Bundesstaat in Mitteleuropa. Er hat 16 Bundesländer und ist als freiheitlich-demokratischer und sozialer Rechtsstaat verfasst. Die 1949 gegründete Bundesrepublik Deutschland stellt die jüngste Ausprägung des 1871 erstmals begründeten deutschen Nationalstaates dar. Bundeshauptstadt und Regierungssitz ist Berlin. Deutschland grenzt an neun Staaten, es hat Anteil an der Nord- und Ostsee im Norden sowie dem Bodensee und den Alpen im Süden. Es liegt in der gemäßigten Klimazone und verfügt über 16 National- und mehr als 100 Naturparks."
    doc = tokenize_documents(
        [text],
        nlp.tokenize_as_words,
        merge_named_entities=merge_named_entities,
        merge_noun_chunks=merge_noun_chunks,
    ).__next__()

    assert ("ein Bundesstaat" in doc) == merge_noun_chunks
    assert ("16 Bundesländer" in doc) == merge_noun_chunks
    assert (
        "Die 1949 gegründete Bundesrepublik Deutschland" in doc
    ) == merge_noun_chunks
    print(doc)
    assert ("Bundesrepublik Deutschland" in doc) == (
        merge_named_entities and not merge_noun_chunks
    )

    if not merge_noun_chunks and not merge_named_entities:
        assert len(doc) == 85


@given(nlp_st.texts)
@settings(deadline=None)
def test_noun_chunks(text: str):
    # the tested function only works for documents tokenized by spacy directly
    doc = list(tokenize_documents([text], nlp.tokenize_as_words))[0]
    chunks = nlp.noun_chunks(doc)

    # property functions must have the same length as the original document
    assert len(chunks) == len(doc)

    # test the tokenization using noun chunks


def test_noun_chunks_without_nouns():
    text = "gehen, laufen, schwimmen"
    doc = list(tokenize_documents([text], nlp.tokenize_as_words))[0]
    chunks = nlp.noun_chunks(doc)

    assert all(chunk is None for chunk in chunks)


def test_noun_chunks_with_nouns():
    text = "Ein hungriger Hund geht in einem schönen See baden"
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


# due to caching, this kind of operation used to fail in a previous version
def test_posterior_merging():
    text = "Ein hungriger Hund geht in einem schönen See baden"
    doc = tokenize_documents([text], nlp.tokenize_as_words).__next__()

    assert len(doc) == len(text.split(" "))
    assert all(token == word for token, word in zip(text.split(" "), doc))

    test_noun_chunks_with_nouns()

    doc = tokenize_documents([text], nlp.tokenize_as_words).__next__()

    assert len(doc) == len(text.split(" "))
    assert all(token == word for token, word in zip(text.split(" "), doc))


@given(nlp_st.texts)
@settings(deadline=None)
def test_get_word_vectors(text: str):
    doc = tokenize_documents([text], nlp.tokenize_as_words).__next__()
    vectors = nlp.get_word_vectors(doc)

    assert len(vectors) == len(doc)


def test_get_word_vectors_with_merging():
    text = "Ein hungriger Hund geht in einem schönen See baden."
    doc = tokenize_documents(
        [text], nlp.tokenize_as_words, merge_noun_chunks=True
    ).__next__()
    vectors = nlp.get_word_vectors(doc)

    assert len(vectors) == len(doc)
