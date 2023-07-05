from typing import Protocol


# On the lowest level, represent documents as tuples of tokens.
# These tokens are considered the most atomic part of the document.
# We choose tuples, rather than generic collections, because
# tuples are hashable and documents should not be mutable.
Document = tuple[str]

# The result of a filter is an immutable set of indices.
Filter_Result = frozenset[int]


class Filter(Protocol):
    """
    Functions that filter tokenized documents contents in some way.

    Filter functions are composed to created pipelines.
    """

    def __call__(self, doc: Document) -> Filter_Result:
        """
        Return an immutable set of token-indices to keep.

        :param doc: The tokenized document to process.
        """
        ...
