#!/usr/bin/env python3
import typing as T
from string import punctuation

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize


def deepest():
    """Find and print the synset with the largest maximum depth along with its
    depth on each of its hyperonym paths.

    Returns:
        None
    """
    max_depth = 0
    max_synset = None
    for synset in wn.all_synsets():
        depth = synset.max_depth()
        if depth >= max_depth:
            max_synset, max_depth = synset, depth
    search = max_synset.hypernym_paths()
    for path in search:
        if len(path) == max_depth + 1:

            # Print from max_depth to 0
            curr_depth = max_depth
            for synset in reversed(path):
                print(curr_depth, synset)
                curr_depth -= 1

            # Alternatively: depth 0 to max_depth
            # for index, synset in enumerate(path):
            #     print(index, synset)


def superdefn(s: str) -> T.List[str]:
    """Get the "superdefinition" of a synset. (Yes, superdefinition is a
    made-up word. All words are made up...)

    We define the superdefinition of a synset to be the list of word tokens,
    here as produced by word_tokenize, in the definitions of the synset, its
    hyperonyms, and its hyponyms.

    Args:
        s (str): The name of the synset to look up

    Returns:
        list of str: The list of word tokens in the superdefinition of s

    Examples:
        >>> superdefn('toughen.v.01')
        ['make', 'tough', 'or', 'tougher', 'gain', 'strength', 'make', 'fit']
    """
    target = wn.synset(s)
    word_tokens = list()
    word_tokens += word_tokenize(target.definition())
    for hyperonym in target.hypernyms():
        word_tokens += word_tokenize(hyperonym.definition())
    for hyponym in target.hyponyms():
        word_tokens += word_tokenize(hyponym.definition())
    return word_tokens


def stop_tokenize(s: str) -> T.List[str]:
    """Word-tokenize and remove stop words and punctuation-only tokens.

    Args:
        s (str): String to tokenize

    Returns:
        list[str]: The non-stopword, non-punctuation tokens in s

    Examples:
        >>> stop_tokenize('The Dance of Eternity, sir!')
        ['Dance', 'Eternity', 'sir']
    """
    tokens = word_tokenize(s)
    tokens = [t for t in tokens if t.lower() not in stopwords.words('english')]
    punc = list(punctuation)
    tokens = [t for t in tokens if t not in punc]
    return tokens


if __name__ == '__main__':
    import doctest
    doctest.testmod()
