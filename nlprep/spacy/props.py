"""
A collection of functions that compute various text properties,
based on the spaCy library and its de_dep_news_trf model in particular.

See https://huggingface.co/spacy/de_dep_news_trf for more details on the model.

To reduce decrease unnecessary calculations,
cache spaCy representations of already processed documents.

Note that even though these functions are defined on
spaCy-specific document representations,
they will actually act on the internal Document representation.
"""
import nlprep.spacy.utils as utils
from nlprep.types import Tokens

import spacy.tokens


def tokenize_as_words(raw_doc: str) -> Tokens:
    """Tokenize a document into its words"""
    return utils.raw_into_property(raw_doc, "text")


def tokenize_as_lemmas(raw_doc: str) -> Tokens:
    """Tokenize a document into its lemmas"""
    return utils.raw_into_property(raw_doc, "lemma_")


@utils.property_from_doc
def get_upos(processed_doc: spacy.tokens.Doc) -> list[str]:
    """The universal POS tags of each token"""
    return [token.pos_ for token in processed_doc]


@utils.property_from_doc
def is_stop(processed_doc: spacy.tokens.Doc) -> list[bool]:
    """Indicators whether each token is a stop word"""
    return [token.is_stop for token in processed_doc]


@utils.property_from_doc
def lemmatize(processed_doc: spacy.tokens.Doc) -> list[str]:
    """The lemmatized version of each token"""
    return [token.lemma_ for token in processed_doc]


@utils.sentencizer_from_doc
def into_sentences(processed_doc: spacy.tokens.Doc) -> list[list[str]]:
    """Split the document by its sentences"""
    return [[token.text for token in sent] for sent in processed_doc.sents]


@utils.sentencizer_from_doc
def into_sentences_lemmatized(
    processed_doc: spacy.tokens.Doc,
) -> list[list[str]]:
    """Split the document by its sentences, with lemmatization"""
    return [[token.lemma_ for token in sent] for sent in processed_doc.sents]


@utils.property_from_doc
def noun_chunks(processed_doc: spacy.tokens.Doc) -> list[int | None]:
    """
    Annotate the given tokens with the noun chunks they belong to.

    If a token does not belong to a noun chunk, it is assigned None.
    NOTE: the given document has to have been tokenized by spaCy!
    """
    chunks = list(processed_doc.noun_chunks)

    if len(chunks) == 0:
        return [None for _ in processed_doc]

    res: list[int | None] = []
    cur_chunk = 0
    chunk_was_visited = [False for _ in chunks]
    for token in processed_doc:
        # all noun chunks have been searched through
        if cur_chunk >= len(chunks):
            res.append(None)
            continue

        if token in chunks[cur_chunk]:
            res.append(cur_chunk)
            chunk_was_visited[cur_chunk] = True
            continue

        if chunk_was_visited[cur_chunk]:
            cur_chunk += 1

        res.append(None)

    return res
