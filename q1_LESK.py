#!/usr/bin/env python3
from collections import Counter
from typing import *

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np
from numpy.linalg import norm

from q0_Base_Functions import stop_tokenize
from wsd import evaluate, load_eval, load_word2vec, WSDToken


def mfs(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Most frequent sense of a word.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. See the WSDToken class in wsd.py
    for the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The most frequent sense for the given word.
    """
    # raise NotImplementedError
    word = sentence[word_index]
    mf_synset = wn.synsets(word.lemma)
    if not mf_synset:   # empty synset list
        return None
    else:
        return mf_synset[0]


def lesk(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Simplified Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    # raise NotImplementedError
    word = sentence[word_index]
    mf_synset = wn.synsets(word.lemma)
    if not mf_synset:  # empty synset list
        return None
    elif len(mf_synset) == 1:
        return mf_synset[0]

    # Add sentence words to context
    context = Counter()
    for word in sentence:
        context.update({word.wordform: 1})

    # LESK
    best_sense = mf_synset[0]
    best_score = 0
    for synset in mf_synset:
        signature = Counter()
        signature.update(stop_tokenize(synset.definition()))    # definition
        for example in synset.examples():   # examples
            signature.update(stop_tokenize(example))
        intersection = signature & context  # takes intersection (min count)
        score = sum(intersection.values())  # cardinality sum
        if score > best_score:
            best_sense = synset
            best_score = score
    return best_sense


def lesk_ext(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    # raise NotImplementedError
    word = sentence[word_index]
    mf_synset = wn.synsets(word.lemma)
    if not mf_synset:  # empty synset list
        return None
    elif len(mf_synset) == 1:
        return mf_synset[0]

    # Add sentence words to context
    context = Counter()
    for word in sentence:
        context.update({word.wordform: 1})

    best_sense = mf_synset[0]
    best_score = 0
    for synset in mf_synset:
        signature = Counter()

        signature_updater(synset, signature)    # def, explanation
        ext_updater(synset, signature)  # def, explanation of all extensions

        intersection = signature & context  # overlap
        score = sum(intersection.values())
        if score > best_score:
            best_sense = synset
            best_score = score
    return best_sense


def ext_updater(synset, signature):
    for hyponym_synset in synset.hyponyms():
        signature_updater(hyponym_synset, signature)

    for member_holonym in synset.member_holonyms():
        signature_updater(member_holonym, signature)
    for part_holonym in synset.part_holonyms():
        signature_updater(part_holonym, signature)
    for sub_holonym in synset.substance_holonyms():
        signature_updater(sub_holonym, signature)

    for member_meronym in synset.member_meronyms():
        signature_updater(member_meronym, signature)
    for part_meronym in synset.part_meronyms():
        signature_updater(part_meronym, signature)
    for substance_meronym in synset.substance_meronyms():
        signature_updater(substance_meronym, signature)


def signature_updater(synset, signature):
    signature.update(stop_tokenize(synset.definition()))
    for example in synset.examples():  # examples
        signature.update(stop_tokenize(example))


def lesk_cos(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    # raise NotImplementedError
    word = sentence[word_index]
    mf_synset = wn.synsets(word.lemma)
    if not mf_synset:  # empty synset list
        return None
    elif len(mf_synset) == 1:
        return mf_synset[0]

    # Add sentence words to context
    context = Counter()
    for word in sentence:
        context.update({word.wordform: 1})

    best_sense = mf_synset[0]
    best_score = 0
    for synset in mf_synset:
        signature = Counter()
        signature_updater(synset, signature)    # def, explanation
        ext_updater(synset, signature)  # def, explanation of all extensions

        # Merge words into one vector. Map indexes to two new vectors
        union = (context | signature).keys()
        union_len = len(union)
        c_vector = [0] * union_len
        s_vector = [0] * union_len

        # map word counts to the new vectors
        for index, word in enumerate(union):
            if word in context.keys():
                c_vector[index] = context[word]
            if word in signature.keys():
                s_vector[index] = signature[word]
        c_vector = np.array(c_vector)
        s_vector = np.array(s_vector)
        norm_value = norm(c_vector) * norm(s_vector)

        # Check we are not dividing by 0
        if norm_value == 0:
            score = 0
        else:
            # cosine similarity calculation
            score = np.dot(c_vector, s_vector) / norm_value
        if score > best_score:
            best_sense = synset
            best_score = score
    return best_sense


def lesk_cos_onesided(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using one-sided cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    # raise NotImplementedError
    word = sentence[word_index]
    mf_synset = wn.synsets(word.lemma)
    if not mf_synset:  # empty synset list
        return None
    elif len(mf_synset) == 1:
        return mf_synset[0]

    # Add sentence words to context
    context = Counter()
    for word in sentence:
        context.update({word.wordform: 1})

    best_sense = mf_synset[0]
    best_score = 0
    for synset in mf_synset:
        signature = Counter()
        signature_updater(synset, signature)
        ext_updater(synset, signature)

        c_vector = list(context.values())   # one-sided vector
        s_vector = [0] * len(context)

        # Map counts to signature vector
        for index, word in enumerate(context.keys()):
            if word in signature.keys():
                s_vector[index] = signature[word]
        c_vector = np.array(c_vector)
        s_vector = np.array(s_vector)
        norm_value = norm(c_vector) * norm(s_vector)
        if norm_value == 0:
            score = 0
        else:
            score = np.dot(c_vector, s_vector) / norm_value
        if score > best_score:
            best_sense = synset
            best_score = score
    return best_sense


def vocab_search(word, vocab, word2vec):

    if word in vocab:   # normal case
        w2v_vocab = vocab[word]
        return word2vec[w2v_vocab]
    else:
        lower_case = word.lower()
        if lower_case in vocab:
            w2v_vocab = vocab[lower_case]  # check lower case
            return word2vec[w2v_vocab]
        else:
            return None     # can't find vector


def check_space_vocab_search(word, vocab, word2vec, length):
    # If it has a space: strip space, replace with underscore
    if ' ' in word:
        new_word = word.replace(" ", "_")
        vector = vocab_search(new_word, vocab, word2vec)
        if vector is not None:
            return vector
        else:
            # if not found: split words, vocab search each, return mean vector
            words = word.split()
            vector = np.zeros(length)
            for w in words:
                result = vocab_search(w, vocab, word2vec)
                if result is None:
                    result = np.zeros(length)
                vector += result
            return vector / len(words)
    else:  # basic vocab search
        vector = vocab_search(word, vocab, word2vec)
        if vector is not None:
            return vector
        else:
            return np.zeros(length)


def lesk_w2v(sentence: Sequence[WSDToken], word_index: int,
             vocab: Mapping[str, int], word2vec: np.ndarray) -> Synset:
    """Extended Lesk algorithm using word2vec-based cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    To look up the vector for a word, first you need to look up the word's
    index in the word2vec matrix, which you can then use to get the specific
    vector. More directly, you can look up a string s using word2vec[vocab[s]].

    To look up the vector for a *single word*, use the following rules:
    * If the word exists in the vocabulary, then return the corresponding
      vector.
    * Otherwise, if the lower-cased version of the word exists in the
      vocabulary, return the corresponding vector for the lower-cased version.
    * Otherwise, return a vector of all zeros. You'll need to ensure that
      this vector has the same dimensions as the word2vec vectors.

    But some wordforms are actually multi-word expressions and contain spaces.
    word2vec can handle multi-word expressions, but uses the underscore
    character to separate words rather than spaces. So, to look up a string
    that has a space in it, use the following rules:
    * If the string has a space in it, replace the space characters with
      underscore characters and then follow the above steps on the new string
      (i.e., try the string as-is, then the lower-cased version if that
      fails), but do not return the zero vector if the lookup fails.
    * If the version with underscores doesn't yield anything, split the
      string into multiple words according to the spaces and look each word
      up individually according to the rules in the above paragraph (i.e.,
      as-is, lower-cased, then zero). Take the mean of the vectors for each
      word and return that.
    Recursion will make for more compact code for these.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.
        vocab (dictionary mapping str to int): The word2vec vocabulary,
            mapping strings to their respective indices in the word2vec array.
        word2vec (np.ndarray): The word2vec word vectors, as a VxD matrix,
            where V is the vocabulary and D is the dimensionality of the word
            vectors.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    word = sentence[word_index]
    mf_synset = wn.synsets(word.lemma)
    if not mf_synset:  # empty synset list
        return None
    elif len(mf_synset) == 1:
        return mf_synset[0]

    # Add sentence words to context
    context = set()
    for word in sentence:
        context.update([word.wordform])

    best_sense = mf_synset[0]
    best_score = 0
    for synset in mf_synset:
        signature = set()
        signature_updater(synset, signature)    # def, explanation
        ext_updater(synset, signature)  # def, explanation of all extensions

        length = 300
        c_vector = np.zeros(length)
        s_vector = np.zeros(length)

        # c_vector and s_vector computation
        for word in context:
            c_vector += check_space_vocab_search(word, vocab, word2vec, length)
        for word in signature:
            s_vector += check_space_vocab_search(word, vocab, word2vec, length)
        c_vector /= len(context)
        s_vector /= len(signature)
        norm_value = norm(c_vector) * norm(s_vector)

        # Check we are not dividing by 0
        if norm_value == 0:
            score = 0
        else:
            # cosine similarity calculation
            score = np.dot(c_vector, s_vector) / norm_value
        if score > best_score:
            best_sense = synset
            best_score = score
    return best_sense


if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()
    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos, lesk_cos_onesided]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
