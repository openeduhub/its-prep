from collections.abc import Sequence, Set

import hypothesis.strategies as st
from its_prep.types import Document, Filter, Property_Function, Tokens

# fundamental types
characters = st.characters(max_codepoint=2047)
texts = st.text(characters)
texts_non_empty = st.text(characters, min_size=1)

# tokens must not be empty
tokens = st.lists(texts_non_empty).map(lambda x: tuple(x))

tokenizers = st.functions(like=lambda text: ..., returns=tokens, pure=True)


@st.composite
def property_funs(draw) -> Property_Function[str]:
    # emulate immutable function behavior
    cache: dict[Tokens, Sequence[str]] = dict()

    def fun(doc: Document) -> Sequence[str]:
        n = len(doc.original_tokens)
        return cache.setdefault(
            doc.original_tokens, draw(st.lists(texts_non_empty, min_size=n, max_size=n))
        )

    return fun


documents = tokens.map(Document.fromtokens)


@st.composite
def subsets(draw, given_set: Set[int]) -> Set[int]:
    # the only subset of an empty set is the empty set itself
    if not given_set:
        return set()

    return given_set & draw(
        st.sets(st.integers(min_value=min(given_set), max_value=max(given_set)))
    )


@st.composite
def documents_with_selections(draw) -> Document:
    doc: Document = draw(documents)
    subset: Set[int] = draw(subsets(doc.selected))
    return doc.sub_doc(subset)


@st.composite
def filters(draw) -> Filter:
    # emulate immutable function behavior
    cache: dict[str, frozenset[int]] = dict()

    def filter_fun(doc: Document) -> Document:
        selected = cache.setdefault(
            doc.original_text,
            draw(
                st.frozensets(
                    st.integers(min_value=0, max_value=len(doc.original_tokens))
                )
            ),
        )
        return doc.sub_doc(selected)

    return filter_fun


@st.composite
def filters_unsafe(draw) -> Filter:
    # emulate immutable function behavior
    cache: dict[Tokens, frozenset[int]] = dict()

    def filter_fun(doc: Document) -> Document:
        selected = cache.setdefault(doc.selected_tokens, draw(subsets(doc.selected)))
        return doc.sub_doc(selected)

    return filter_fun
