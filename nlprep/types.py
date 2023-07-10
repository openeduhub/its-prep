from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Set
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, eq=True)
class Document:
    original_text: str
    original_tokens: tuple[str, ...]
    selected: frozenset[int]

    def __repr__(self) -> str:
        return self.tokens.__repr__()

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(self.original_tokens[index] for index in self.selected)

    @classmethod
    def fromtext(
        cls, text: str, tokenize_fun: Callable[[str], tuple[str, ...]]
    ) -> Document:
        tokens = tokenize_fun(text)
        return Document(
            original_text=text,
            original_tokens=tokens,
            selected=frozenset(range(len(tokens))),
        )

    @classmethod
    def fromtokens(cls, __iterable: Iterable[str]) -> Document:
        tokens = tuple(__iterable)
        text = " ".join(tokens)
        return Document(
            original_text=text,
            original_tokens=tokens,
            selected=frozenset(range(len(tokens))),
        )

    def sub_doc(self, selected_indices: Set[int]) -> Document:
        return Document(
            original_text=self.original_text,
            original_tokens=self.original_tokens,
            selected=self.selected & selected_indices,
        )


class Filter(Protocol):
    """
    Functions that filter tokenized documents contents in some way.

    Filter functions are composed to created pipelines.
    """

    def __call__(self, doc: Document) -> Document:
        """
        Return a sub-document of the given document.
        These should be created using the doc.sub_doc method.

        :param doc: The document to process.
        """
        ...


Pipeline = Collection[Filter]


class Pipeline_Generator(Protocol):
    """
    Functions that generate pre-processing pipeline, given a document corpus.

    The reason why we supply the document corpus here is because some filters
    may require an initial analysis of all documents,
    e.g. in order to filter by the document frequency of lemmatized tokens.
    """

    def __call__(self, docs: Collection[Document], **kwargs) -> Pipeline:
        ...
