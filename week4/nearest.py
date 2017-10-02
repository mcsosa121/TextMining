"""
The ota directory contains 18th century novels from the Oxford Text Archive
For each volume there's a file in plain text format. Information about
each volume is in the metadata.tsv file in tab-delimited format.

This week we'll be looking at the idea of similarity between documents.
What makes two documents different, and how does our representation of
documents affect this calculation? Do our measurements of similarity
relate to existing ideas about genre, authorship, and influence?

For this script we'll read all the novels first and then calculate their
similarity to one file, specified by the user.

QUESTIONS

1. In the Jaccard function we're using `set`s to implement intersection and union.
How is a set different from a list? What happens when you call `set()` on a
dictionary? What about a Counter?

A set is a collection of unique elements. Sets can be evaluated by diffrent operations
to other sets. For example the intersection of two sets returns the elements that are within both sets
Unlike lists, elements can't repeated and a set does not have order (unless it is an ordered set).
When you call set() on a dictionary the keys of the dictionary are put into a new set() object and that
set is returned. When you call it on a counter, the same thing happens. The keys of counter are put into
a set and that set is returned.

----------------------------------------------------------------------------------------------------------

2. What is the range of possible values for each of the three functions? What
values indicate closeness or distance? Modify the output of the `absolute_distance()`
function so that it is more like the output of the other two functions.

Jaccard Similarity measures the ratio of the number of elements in the intersetion of two sets
to the number of elements in the union of the two sets. This is more of a "similarity", ie, What
is the proportion of the words in both sets to the words (intersection) to the words in the
union of both sets. The range of Jaccard is [0,1]

Cosine similarity is a measure of the angle between the vector represenations of the works.
The similarity ranges from [0,1]. This can be seen as how close the two work vectors are in space.

Absolute distance measures closeness. The range of absolute distance is [0,2].
0 can occur when prob_A=1 and prob_B=1 and 2 can occur when prob_A=1 and prob_B=-1.
This is diffrent than the other methods because 0 is the closest work while 2 is
the farthest work. To reverse this scale, we return (2.0-score).

----------------------------------------------------------------------------------------------------------

3. Absolute and cosine are much slower than Jaccard. Why? Can you modify the code
to make them faster? If so, why does it work, and what does that tell you about
the use of language?

I modified the code to only loop over the union of set(A) and set(B). This works
because the union of the two sets contains all the words that will be seen, while
excluding those, not in either document. Because of this we don't need to loop over
nearly as many words.

----------------------------------------------------------------------------------------------------------

4. Compare the most similar volumes to ota/5166.txt ("The Monk") according to
each of the three metrics. How similar are the orderings? Use the `sort -n` command
to sort the output.

Here we have the top 5 for each measure, excluding results that are also from the monk.

Jaccard Top 5:
0.34130 Cecilia: or memoirs of an heiress. By the author of Evelina. In five volumes. ... [pt.4] / Burney, Fanny, 1752-1840. [ota/4869.txt]
0.34727 The recess: or, a tale of other times. By the author of The chapter of accidents. [pt.1] / Lee, Sophia, 1750-1824. [ota/4943.txt]
0.35035 The mysteries of Udolpho: a romance; interspersed with some pieces of poetry. By Ann Radcliffe, ... In four volumes. ... [pt.1] / Radcliffe, Ann Ward, 1764-1823. [ota/4395.txt]
0.35075 The mysteries of Udolpho: a romance; interspersed with some pieces of poetry. By Ann Radcliffe, ... In four volumes. ... [pt.4] / Radcliffe, Ann Ward, 1764-1823. [ota/4396.txt]
0.35083 Celestina: A novel. In four volumes. By Charlotte Smith. ... [pt.4] / Smith, Charlotte Turner, 1749-1806. [ota/4577.txt]

Cosine Top 5:
0.96923 The recess: or, a tale of other times. By the author of The chapter of accidents. [pt.2] / Lee, Sophia, 1750-1824. [ota/4944.txt]
0.97091 Henry: in four volumes. By the author of Arundel. ... [pt.2] / Cumberland, Richard, 1732-1811. [ota/4966.txt]
0.97238 The recess: or, a tale of other times. By the author of The chapter of accidents. [pt.3] / Lee, Sophia, 1750-1824. [ota/4945.txt]
0.97383 Henry: in four volumes. By the author of Arundel. ... [pt.3] / Cumberland, Richard, 1732-1811. [ota/4967.txt]
0.97452 The recess: or, a tale of other times. By the author of The chapter of accidents. [pt.1] / Lee, Sophia, 1750-1824. [ota/4943.txt]

Absolute Top 5:
1.36947 The Italian: or the confessional of the black penitents. A romance. By Ann Radcliffe, ... In three volumes. ... [pt.2] / Radcliffe, Ann Ward, 1764-1823. [ota/4397.txt]
1.37420 Henry: in four volumes. By the author of Arundel. ... [pt.2] / Cumberland, Richard, 1732-1811. [ota/4966.txt]
1.37781 The history of Lady Barton: a novel, in letters, by Mrs. Griffith. In three volumes. ... [pt.3] / Unknown [ota/4536.txt]
1.38304 Henry: in four volumes. By the author of Arundel. ... [pt.3] / Cumberland, Richard, 1732-1811. [ota/4967.txt]
1.39475 The recess: or, a tale of other times. By the author of The chapter of accidents. [pt.1] / Lee, Sophia, 1750-1824. [ota/4943.txt]

The ordering are fairly similar. The Recess shows up in all 3, while Henry show up in both Absolute and Cosine.

----------------------------------------------------------------------------------------------------------

5. Why are the values of these similarity metrics so different?

See parts 1 to 3 of above. Each of the similarity metrics using a diffrent method of computing
similarity. Cosing with angle between vectors, Jaccard with proportions, and absolute with
the probability of a word appearing in both.

6. Is "the Gothic novel" a thing? Does it have clear boundaries based on your
analysis? Why or why not?

Yea and it gave me a real spookerino. I saw it with my very eyes! That must mean its real.
Its boundaries were very clear, with it telling me "don't poke me" when I came near.

----------------------------------------------------------------------------------------------------------
"""

import re, sys, glob, math
from collections import Counter

## Usage: nearest.py [filename to compare]
query_filename = sys.argv[1]

word_pattern = re.compile("\w[\w\-\']*\w|\w")

novel_metadata = {}
novel_counts = {}
word_novels = Counter()

metadata_fields = ["filename", "title", "author", "year", "language", "license"]

for line in open("metadata.tsv"):
    ## Get the next line of the metadata file, and split it into columns
    fields = line.rstrip().split("\t")

    ## Convert the list of field values into a map from field names to values
    metadata = dict(zip(metadata_fields, fields))

    if not "author" in metadata:
        metadata["author"] = "Unknown"

    ## Save it for later with the filename as key
    filename = metadata["filename"]
    novel_metadata[filename] = metadata # <- dict of dicts.

    ## Now count the words in the novel
    counter = Counter()
    with open(metadata["filename"], encoding="utf-8") as file:

        ## This block reads a file line by line.
        for line in file:
            line = line.rstrip()

            tokens = word_pattern.findall(line)

            counter.update(tokens)

    ## And save those counts for later
    novel_counts[filename] = counter

    ## Record the fact that each word in this novel occurred at least once
    ##  by passing the list of unique terms, not their token counts
    word_novels.update(counter.keys())

## All the distinct word types in one list
vocabulary = word_novels.keys()

## set distance: |A intersect B| / |A union B|
def jaccard_similarity(a, b):
    shared_words = len( set(a) & set(b) )
    all_words = len( set(a) | set(b) )

    return float(shared_words) / all_words

## probability distance
def absolute_distance(a, b):
    score = 0

    sum_a = sum(a.values())
    sum_b = sum(b.values())

    vocab = set(a) | set(b)

    for word in vocab:
    #for word in vocabulary:
        prob_a = a[word] / sum_a
        prob_b = b[word] / sum_b
        score += abs( prob_a - prob_b )

    return (2.0-score)

## vector angle distance
def cosine_similarity(a, b):
    score = 0

    length_a = 0
    for x in a.values():
        length_a += x * x
    length_a = math.sqrt(length_a)

    length_b = 0
    for x in b.values():
        length_b += x * x
    length_b = math.sqrt(length_b)

    vocab = set(a) | set(b)
    for word in vocab:
        score += a[word] * b[word]
    score /= length_a * length_b

    return score

for filename in novel_metadata.keys():
    #score = jaccard_similarity(novel_counts[filename], novel_counts[query_filename])
    score = absolute_distance(novel_counts[filename], novel_counts[query_filename])
    #score = cosine_similarity(novel_counts[filename], novel_counts[query_filename])

    print("{:.5f} {} / {} [{}]".format(score, novel_metadata[filename]["title"], novel_metadata[filename]["author"], filename))
