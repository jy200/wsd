## q1_LESK models:
- mfs: 41.6%
- lesk: 39.3%
- lesk_ext: 45.9%
- lesk_cos: 38.3%
- lesk_cos_onesided: 44.0%
- lesk_w2v: 47.9%

lesk_cos_onesided performed better than lesk_cos by more than 5%. This difference in accuracy is because the vectors in lesk_cos include both context and signature words while lesk_cos_onesided only includes the context words.

For example, say we wanted to predict the correct sense for "art" in the context sentence "The art of
change-ringing is peculiar to the English language, and like most English peculiarities, unintelligible to
the rest of the world." Well, lesk_cos would have wordforms like "computer”, “printmaking”, “parts”,
and “software" in their vectors (due to signature) which are unrelated to our context sentence, while
lesk_cos_onesided would only have wordforms related to the context sentence like "English", "changeringing,"
"peculiar". As a result, lesk_cos_onesided vectors are more relevant and similar to the context
sentence while lesk_cos vectors have some differences resulting from the inclusion of senses with nonrelevant
contexts in the signature set’s definitions and explanations.

Since lesk_cos_onesided vectors are more relevant to the context sentence, the function predicts the
correct sense better on average compared to lesk_cos.

## q2_BERT model:
### Accuracy: ~49%
When the run_bert function is called to tokenize the batch for BERT, it pads the tokenized sentences
up to the longest batch sentence. Hence, mixing short and long sentences increases run time due to the
relative increased padding. In comparison, there is minimal padding when sentences of similar length of
batched.



Our code model was trained on certain words in certain contexts, e.g., our corpus in Q2
gather_sense_vectors. If we were to throw an arbitrary sentence with words not in the dictionary
returned by gather_sense_vectors, it is likely to predict the wrong sense (Since there are no sense
vectors, Q2 uses MFS which only assumes the most frequent sense. Bad if our sense is used less
frequently).

Of course, depending on the context of the arbitrary sentence, we would still have problems even if it
was in the dictionary returned by gather_sense_vectors. This is because it was trained on a certain
corpus, so the context might be entirely different, resulting in a misclassification once again (In other
words, the predicted senses will be similar to the ones in the corpus. Likely to result in a misclassification
if sentence uses the word in contexts very different than the ones in the corpus).
