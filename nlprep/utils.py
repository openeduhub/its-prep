from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Optional, TypeVar, Generic
from pathlib import Path
import pickle
import spacy
import spacy.tokens

T = TypeVar("T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def nest(fun: Callable[[T], T], n: int) -> Callable[[T], T]:
    """Helper function to apply a function onto itself n times."""

    def nested_fun(x: T) -> T:
        for _ in range(n):
            x = fun(x)

        return x

    return nested_fun


class Keyed_defaultdict(defaultdict, Generic[_KT, _VT]):
    """A custom version defaultdict that supports keyed factories"""

    default_factory: Callable[[_KT], _VT]

    def __init__(self, default_factory: Callable[[_KT], _VT]):
        self.default_factory = default_factory

    def __missing__(self, __key: _KT) -> _VT:
        """Override the missing method in order to pass the looked up key to the factory"""
        value = self.default_factory(__key)
        self[__key] = value
        return value

    @classmethod
    def from_file(
        cls, default_factory: Callable[[_KT], _VT], path: Path
    ) -> Keyed_defaultdict:
        """A new keyed defaultdict with data from the given file path."""
        with open(path, "rb+") as f:
            # only load the underlying data
            data = pickle.load(f)

        obj = cls(default_factory)
        obj.update(data)
        return obj

    def save(self, path: Path) -> None:
        """
        Save the underlying data to the given file path.

        Note that the given default factory is NOT saved -- this is because
        anonymous or local functions cannot be exported / imported here.
        """
        with open(path, "wb+") as f:
            # only dump the underlying data
            pickle.dump(dict(self), f)


class Spacy_defaultdict(Keyed_defaultdict[_KT, spacy.tokens.Doc]):
    @classmethod
    def from_file(
        cls,
        default_factory: Callable[[_KT], spacy.tokens.Doc],
        keys_path: Path,
        docs_path: Path,
        vocab: spacy.vocab.Vocab,
    ) -> Spacy_defaultdict:
        # load the underlying keys
        with open(keys_path, "rb") as f:
            keys: Iterable[_KT] = pickle.load(f)

        # load the underlying documents
        docbin = spacy.tokens.DocBin()
        docbin.from_disk(docs_path)
        docs = docbin.get_docs(vocab)

        # combine them into the new cache
        obj = cls(default_factory)
        data = {key: doc for key, doc in zip(keys, docs)}
        obj.update(data)
        return obj

    def save(self, keys_path: Path, docs_path: Path) -> None:
        """
        Save the underlying keys and documents to the given file paths.

        Note that the given default factory is NOT saved -- this is because
        anonymous or local functions cannot be exported / imported here.
        """
        # dump the keys
        with open(keys_path, "wb+") as f:
            pickle.dump(list(self.keys()), f)

        # dump the documents, using the DocBin utility from spacy
        spacy.tokens.DocBin(docs=self.values()).to_disk(docs_path)
