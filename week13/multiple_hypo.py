"""
More Dunning's log-likelihood test, Multiple hypotheses.

Can an observation be significant but not convincing?

In text mining we often count things and compare 
proportions. Frequently we can count the same thing
in different contexts, like a word in two different
novels. If we want to make claims about differences
between contexts, we need to know whether an observed
difference could have arisen by random chance.
We can use Ted Dunning's
G^2 metric as an indicator of significance, but how does 
the number of words we test affect our evaluation of 
significance?

First, refresh your memory about Dunning's G^2:

1. As we did last week, try a few different proportions.
Record the output of dunning_score(...) here.

>>> dunning_score(5,10,100,100)
1.8341419597039819
>>> dunning_score(5,10,1000,1000)
1.7115848397103148
>>> dunning_score(51,49,100,100)
0.08000533418685407
>>> dunning_score(20,1,100,100)
23.092094165233583
>>> dunning_score(10,1,100,100)
8.97597737584663


2. What score do you think indicates a difference that you
would consider convincing evidence that two observed
proportions are really different?

If over 100 games a player has more than 95% of games then they are def better.
People say score of ~1.3


Now let's look at the difference between two Greek
historians, Herodotus and Thucydides.

3. Run `print_nicely(word_scores[:50])`. Copy the output here.
How do the observed Dunning G scores compare to your expectation?

>>> print_nicely(word_scores[:50])
975.544 3842    891     he
830.17  1782    2780    their
675.685 446     1146    Athenians
639.223 641     1       thou
613.622 227     796     you
594.931 625     4       Persians
519.779 67      464     allies
499.757 2656    744     is
469.337 1079    147     then
454.524 1534    314     when
441.163 1529    320     I
428.273 144     534     our
380.554 1306    271     these
367.636 636     54      king
363.908 2853    963     this
357.89  8       228     Syracusans
354.387 3       208     heavy
346.41  776     102     said
340.395 2       196     infantry
331.248 158     481     war
314.37  16      226     Meanwhile
312.682 350     5       Then
312.525 60      316     Athenian
309.525 95      372     your
299.388 569     929     The
284.383 685     99      say
277.804 116     380     Athens
272.728 1658    498     him
253.453 32      225     Peloponnesians
240.513 430     39      Now
238.842 464     49      things
228.34  40      224     town
224.623 240     2       Scythians
220.497 250     4       Egyptians
218.329 266     7       Xerxes
210.086 84      282     enemy
209.906 418     46      me
204.742 234     4       thy
200.843 96      292     could
198.844 228     4       namely
195.521 253     9       forth
189.968 579     108     Hellenes
189.632 197     1       Croesus
188.838 279     16      Egypt
184.549 28      173     Sicily
183.159 511     732     we
181.642 204     3       Cyrus
177.675 194     395     without
176.974 125     311     country
164.97  581     123     man

4. What are the most "surprising" words according to 
the Dunning score? Search through the documents for examples.
What did you learn about the content and the style of these
two historians?

We're looking at about 5000 distinct tests, one for each
word. Should we be worried? Let's simulate random
word distributions.

5. Now create two lists, `fake_herodotus` and `fake_thucydides`
using the `shuffle_lists()` function with the `herodotus_tokens`
and `thucydides_tokens` lists as input. Record the length of
all four lists here, and confirm that the new ones have
the same length as the original `_tokens` lists.

6. Now use the "fake" token lists to create `fake_scores`.
Use `print_nicely()` and array slices (ie [:50], etc) to look
at the range of Dunning scores. How do the "most significant" 
scores compare to your expectations about significance?
Would you have been fooled if you didn't realize that
these results were random?

7. Thucydides writes "for it is the habit of humans to trust
the things they desire to unexamined hope, but to confront
the things they reject with the full force of reason." 
How is this relevant to our discussions?

"""

from collections import Counter
import math, re, random

word_pattern = re.compile("\w[\w\-\']*\w|\w")

thucydides_tokens = []
herodotus_tokens = []

scored_words = []


### Evaluate the "surprise factor" of two proportions that are expressed as counts.
###  ie x1 "heads" out of n1 flips.
def dunning_score(x1, x2, n1, n2):
    p1 = float(x1) / n1
    p2 = float(x2) / n2
    p = float(x1 + x2) / (n1 + n2)

    return -2 * ( x1 * math.log(p / p1) + (n1 - x1) * math.log((1 - p)/(1 - p1)) + 
                  x2 * math.log(p / p2) + (n2 - x2) * math.log((1 - p)/(1 - p2)) )

with open("thucydides.txt") as thucydides:
    for line in thucydides:
        thucydides_tokens.extend(word_pattern.findall(line))

with open("herodotus.txt") as herodotus:
    for line in herodotus:
        herodotus_tokens.extend(word_pattern.findall(line))


def score_differences(a, b):
    a_counter = Counter(a)
    b_counter = Counter(b)

    a_length = len(a)
    b_length = len(b)
    vocabulary = a_counter.keys() & b_counter.keys()

    # list of scores for both
    scored_words = []

    for w in vocabulary:
        a_n = a_counter[w]
        b_n = b_counter[w]

        ## Create a tuple containing information about each word
        g_score = dunning_score(a_n, b_n, a_length, b_length)
        scored_words.append( (round(g_score, 3), a_n, b_n, w) )
        scored_words.sort(reverse = True)

    return scored_words


def shuffle_lists(a, b):
    a_length = len(a)
    b_length = len(b)

    merged = list(a)
    merged.extend(b)
    random.shuffle(merged)

    return (merged[:a_length], merged[a_length:])


def print_nicely(scores):
    for word_info in scores:
        print("{}\t{}\t{}\t{}".format(word_info[0], word_info[1], word_info[2], word_info[3]))

word_scores = score_differences(herodotus_tokens, thucydides_tokens)
