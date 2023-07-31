from collections.abc import Collection, Callable, Set
from typing import TypeVar

from numpy import select
from nlprep.types import Document, Filter
import hypothesis.strategies as st

# only generate UTF-8
texts = st.text(st.characters(blacklist_characters=["\ud800"]))

tokens = st.lists(texts).map(lambda x: tuple(x))

tokenizers = st.functions(like=lambda text: ..., returns=tokens, pure=True)

property_funs = st.functions(like=lambda doc: ..., returns=tokens, pure=True)

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
def filters(draw):
    cache = dict()

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
def filters_unsafe(draw):
    cache = dict()

    def filter_fun(doc: Document) -> Document:
        selected = cache.setdefault(doc, draw(subsets(doc.selected)))
        return doc.sub_doc(selected)

    return filter_fun
