from typing import Any, Collection, Protocol


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


Pipeline = Collection[Filter]


class Pipeline_Generator(Protocol):
    """
    Functions that generate pre-processing pipeline, given a document corpus.

    The reason why we supply the document corpus here is because some filters
    may require an initial analysis of all documents,
    e.g. in order to filter by the document frequency of lemmatized tokens.

    :return: A tuple of collections of first unsafe, then safe filters.
             Unsafe filters need the document to be processed to be 'intact',
             e.g. in order to contextualize tokens or to process sentences.
             Safe filters are perfectly valid to apply on documents
             that have already been filtered.
    """

    def __call__(
        self, docs: Collection[Document], **kwargs
    ) -> tuple[Pipeline, Pipeline]:
        ...
