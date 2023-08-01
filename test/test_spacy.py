from collections.abc import Callable, Collection
from nlprep.types import Document, Property_Function, Tokens
from test.strategies import documents, texts

import nlprep.spacy.props as nlp
from hypothesis import given
from hypothesis import strategies as st


@given(texts, st.sampled_from([nlp.tokenize_as_lemmas, nlp.tokenize_as_words]))
def test_tokenizers_do_something(text: str, tokenizer: Callable[[str], Tokens]):
    result = tokenizer(text)

    if text:
        assert result


@given(documents, st.sampled_from([nlp.get_upos, nlp.lemmatize, nlp.is_stop]))
def test_properties_have_correct_len(doc: Document, fun: Property_Function):
    result = fun(doc)
    assert len(result) == len(doc.original_tokens)


@given(texts, st.sampled_from([nlp.into_sentences, nlp.into_sentences_lemmatized]))
def test_splitters_have_correct_len(
    text: str, tokenizer: Callable[[Document], Collection[Collection]]
):
    doc = Document.fromtokens(nlp.tokenize_as_words(text))
    split_result = tokenizer(doc)
    assert sum(len(result) for result in split_result) == len(doc.original_tokens)
