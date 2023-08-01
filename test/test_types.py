from collections.abc import Callable, Set
from nlprep.types import Document, Tokens
from test.strategies import documents, texts, tokenizers, tokens
from hypothesis import given, strategies as st


@given(texts, tokenizers)
def test_document_from_text(text: str, tokenizer: Callable[[str], Tokens]):
    tokens = tokenizer(text)
    doc = Document.fromtext(text=text, tokenize_fun=tokenizer)

    assert doc.original_text == text
    assert doc.original_tokens == tokens
    assert doc.selected == set(range(len(tokens)))
    assert doc.selected_tokens == tokens


@given(tokens)
def test_document_from_tokens(tokens: Tokens):
    doc = Document.fromtokens(tokens)

    assert doc.original_tokens == tokens
    assert doc.selected_tokens == tokens
    assert doc.selected == set(range(len(tokens)))


@given(documents, st.sets(st.integers(min_value=0)))
def test_document_sub_doc(doc: Document, index_set: Set[int]):
    result = doc.sub_doc(index_set)

    assert result.selected == result.selected & index_set

    expected_tokens = [
        token for index, token in enumerate(doc.selected_tokens) if index in index_set
    ]

    assert len(result.selected_tokens) == len(expected_tokens)

    for result_token in result.selected_tokens:
        assert result_token in expected_tokens

    for expected_token in expected_tokens:
        assert expected_token in result.selected_tokens
