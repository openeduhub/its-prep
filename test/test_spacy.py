from collections.abc import Callable, Collection
from nlprep.types import Document, Property_Function, Split_Function, Tokens
import test.strategies as nlp_st

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
