from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Sequence, Set
from dataclasses import dataclass
from typing import Protocol, TypeVar

import py3langid as langid

Tokens = tuple[str, ...]


@dataclass(frozen=True, eq=True)
class Document:
    original_text: str
    original_tokens: Tokens
    selected: frozenset[int]
    language: str

    @property
    def selected_tokens(self) -> Tokens:
        return tuple(self.original_tokens[index] for index in self.selected)

    @classmethod
    def make(
        cls,
        original_text: str,
        original_tokens: Tokens,
        selected: Iterable[int],
        language: str | None = None,
    ) -> Document:
        if language is None:
            language, _ = langid.classify(original_text)
            assert isinstance(language, str)

        return Document(
            original_text=original_text,
            original_tokens=original_tokens,
            selected=frozenset(selected),
            language=language,
        )

    @classmethod
    def fromtext(cls, text: str, tokenize_fun: Callable[[str], Tokens]) -> Document:
        tokens = tokenize_fun(text)
        return Document.make(
            original_text=text,
            original_tokens=tokens,
            selected=range(len(tokens)),
        )

    @classmethod
    def fromtokens(cls, __iterable: Iterable[str]) -> Document:
        tokens = Tokens(__iterable)
        text = " ".join(tokens)
        return Document.make(
            original_text=text,
            original_tokens=tokens,
            selected=range(len(tokens)),
        )

    def sub_doc(self, selected_indices: Set[int]) -> Document:
        return Document(
            original_text=self.original_text,
            original_tokens=self.original_tokens,
            selected=self.selected & selected_indices,
            language=self.language,
        )

    # a document is a Collection over its selected tokens
    def __iter__(self) -> Iterator[str]:
        return self.selected_tokens.__iter__()

    def __len__(self) -> int:
        return len(self.selected_tokens)

    def __contains__(self, obj) -> bool:
        return obj in self.selected_tokens

    def __repr__(self) -> str:
        return self.selected_tokens.__repr__()


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


Pipeline = Sequence[Filter]

Property = TypeVar("Property", covariant=True)


class Property_Function(Protocol[Property]):
    """
    Functions that compute some property for the tokens of the document.
    """

    def __call__(self, doc: Document) -> Sequence[Property]:
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

    def __call__(self, doc: Document) -> Sequence[Sequence[Property]]:
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
