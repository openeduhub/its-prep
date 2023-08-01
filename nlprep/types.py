from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Set
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

Tokens = tuple[str, ...]


@dataclass(frozen=True, eq=True)
class Document:
    original_text: str
    original_tokens: Tokens
    selected: frozenset[int]

    def __repr__(self) -> str:
        return self.selected_tokens.__repr__()

    @property
    def selected_tokens(self) -> Tokens:
        return tuple(self.original_tokens[index] for index in self.selected)

    @classmethod
    def fromtext(cls, text: str, tokenize_fun: Callable[[str], Tokens]) -> Document:
        tokens = tokenize_fun(text)
        return Document(
            original_text=text,
            original_tokens=tokens,
            selected=frozenset(range(len(tokens))),
        )

    @classmethod
    def fromtokens(cls, __iterable: Iterable[str]) -> Document:
        tokens = Tokens(__iterable)
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

Property = TypeVar("Property")


class Property_Function(Protocol[Property]):
    """
    Functions that compute some property for the tokens of the document.
    """

    def __call__(self, doc: Document) -> Collection[Property]:
        """
        Return the property of each *original* token in the document.

        I.e. len(result) == len(doc.original_tokens)
        """
        ...


class Split_Function(Protocol[Property]):
    """
    Functions that compute some property for the tokens of the document,
    splitting them into nested collections.

    Example: a function that splits a document into its sentences.
    """

    def __call__(self, doc: Document) -> Collection[Collection[Property]]:
        """
        Return the property of each *original* token in the document,
        organized by the split semantic.

        I.e. the sum of the lengths of the nested collections
             == len(original_tokens)
        """
        ...


class Pipeline_Generator(Protocol):
    """
    Functions that generate pre-processing pipeline, given a document corpus.

    The reason why we supply the document corpus here is because some filters
    may require an initial analysis of all documents,
    e.g. in order to filter by the document frequency of lemmatized tokens.
    """

    def __call__(self, docs: Collection[Document], **kwargs) -> Pipeline:
        ...
