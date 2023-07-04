from typing import Protocol


# On the lowest level, represent documents as collections of tokens.
# These tokens are considered the most atomic part of the document.
Document = tuple[str]


class Filter(Protocol):
    """
    Functions that filter tokenized documents contents in some way.

    Filter functions are composed to created pipelines.
    """

    def __call__(self, doc: Document) -> frozenset[int]:
        """
        Return an immutable set of token-indices to keep.

        :param doc: The tokenized document to process.
        """
        ...
