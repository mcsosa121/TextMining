# comparing text to set of texts
"""
In this script we'll be examining another similarity measure: Burrows's delta. Unlike
the techniques in nearest.py, Burrows's delta measures the similarity of a text with
 a group of texts.

For this script we'll read all of the ota novels and identify the overall author
groups. Then, we'll calculate the similarity between these author groups and one file,
specified by the user.

QUESTIONS

1. How many authors are there? What problems exist with this author set? Modify how
author metadata is interpreted within the code.

There are 32 unique authors, 34+ if you count the two or more unknown authors in the
set.One of these works with Unknwn author is "The fair Hibernian"
which thanks to some google searching we can reveal to be Anthony Davidson.
https://goo.gl/85i1Qa (This is a legit link, I just had to use a url shortner)
However some authors even when using uniq, are repeated due to non word characters.
We can parse the output with regex "([\w\S ]+)(?<=[^.\n])"
example command is "cut -f 3 metadata.tsv | sort | grep -Po "([\w\S ]+)(?<=[^.\n])" | uniq"
I also implemented this regex into the code so that authors would not be repeated due
to random characters like a period.

exampe:
        mauthor = metadata["author"]
        author = author_pattern.findall(mauthor)[0]

----------------------------------------------------------------------------------------------------------

2. What is the range of Burrows's delta? How does it differ from previous measures in
nearest.py?

Since it uses absoulte value, delta can't be less than 0. So the bounds are [0,inf].
Burrows measure is the following "The mean of the absolute differences between
the z-scores for a set of word-variables in a given text-group and the
z-scores for the same set of word-variables in a target text."
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.842.4317&rep=rep1&type=pdf (Interpreting Burrows
Delta paper)


----------------------------------------------------------------------------------------------------------

3. How is a numpy array different than a list? Why use numpy arrays instead of lists?

Numppy arrays are diffrent than lists in that they are more akin to actual matrices
as opposed to using a list of lists (Note that numpy has a numpy matrix data structure
but this is strictly 2 dimensional while numpy arrays can be N-dimensional).
With numpy arrays you have a lot of diffrent operations you can perform,
that if you were using default python lists you would
have to implement yourself. Examples of this include matrix multiplication, determinant,
eigenvalues, and more linear algebra operations.

----------------------------------------------------------------------------------------------------------

4. What are the three most likely authors of ota/5166.txt ("The Monk" by Matthew Gregory)
according to Burrows's delta? What's the smallest number of most frequent words that can
be used to correctly identify the author of ota/5166.txt? Use the `sort -n` command to
sort the output.

These are the three most likely authors of "The Monk". So Burrows Delta predicts correctly.
(0.54 Lewis, M. G. (Matthew Gregory), 1775-1818)**
0.80 Lennox, Charlotte, ca. 1729-1804
0.82 Radcliffe, Ann Ward, 1764-1823
Burrows delta works using a certain number of most frequent words. By changing this amount
the prediction can become less accurate and even wrong at some point. I changed the amount
from 300 all the way down to 14 and it still predicted Gregory to be the most likely author.
Finally at 13 it gave a wrong prediction. So the smallest number of words needed to
identify the monk is 14.

----------------------------------------------------------------------------------------------------------

5. Why is Burrows's delta so accurate for ota/4855.txt ("The castle of Otranto")?

Burrows Delta is so accurate for "The castle of Otranto" because of the fact that the (I'm assuming)
main characters of the play come up so often. These characters are "Manfred" and "Isabella" (Shout
out to Hippolita though). Because of this, Burrows delta only needs a minimum of 2 words to correctly predict
the author for "The castle of Otranto". Going to 1 word we find that Burrows Delta predicts Laurence Sterne
to be the author. None of Sterne's works contains Manfred or Isabella. However, techically the most frequent
word in "The castle of Otranto" is "said" which is also very similar in ora/3028.txt.

----------------------------------------------------------------------------------------------------------

6. Can word sets other than the most frequent be used with Burrows's delta to identify
authorship? Do different word frequency ranges describe different aspects of a text?

Yes. In http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.71.8771&rep=rep1&type=pdf
the authors reformulate Burrows Delta in terms of other probability distributions and test them.
Diffrent word frequency ranges describe diffrent distances. For example normally, Burrows delat
uses an almost manhatten like distance on a Laplace distribution. However, using the Normal distribution,
the distance is euclidian.

----------------------------------------------------------------------------------------------------------

7. Are there texts that look very similar according to cosine similarity (e.g. Gothic) that
have very different author signatures according to Burrows's delta?

Yes these texts can exist. The word vectors may be very close in space (cosine similarity) but their frequencies
may be very diffrent, so Burrows Delta would not classify them as being close.

----------------------------------------------------------------------------------------------------------

"""

import re, sys
import numpy as np
from collections import Counter

## Usage: nearest.py [filename to compare]
query_filename = sys.argv[1]

word_pattern = re.compile("\w[\w\-\']*\w|\w")
author_pattern = re.compile("([\w\S ]+)(?<=[^.\n])")

novel_metadata = {} # <- dict of dicts
novel_counts = {} # <- dict of Counters
author_counts = {} # <- dict of Counters
word_novels = Counter()

metadata_fields = ["filename", "title", "author", "year", "language", "license"]

for line in open("metadata.tsv"):
    ## Get the next line of the metadata file, and split it into columns
    fields = line.rstrip().split("\t")

    ## Convert the list of field values into a map from field names to values
    metadata = dict(zip(metadata_fields, fields))


    if not "author" in metadata:
        metadata["author"] = "Unknown"
    else:
        mauthor = metadata["author"]
        mauthor= author_pattern.findall(mauthor)[0]

    ## Save it for later with the filename as key
    filename = metadata["filename"]
    novel_metadata[filename] = metadata

    ## Now count the words in the novel
    counter = Counter()
    with open(filename, encoding="utf-8") as file:

        ## This block reads a file line by line.
        for line in file:
            line = line.rstrip()

            tokens = word_pattern.findall(line)

            counter.update(tokens)

    ## And save those counts for later
    novel_counts[filename] = counter

    ## Update author subset counts if current file is not the query file
    if filename != query_filename:
        mauthor = metadata["author"]
        author = author_pattern.findall(mauthor)[0]
        author_counts[author] = author_counts.get(author, Counter()) + counter

    ## Record the fact that each word in this novel occurred at least once
    ## by passing the list of unique terms, not their token counts
    word_novels.update(counter.keys())

## All the distinct word types in one list
vocabulary = list(word_novels.keys())

## Get author frequencies
author_freqs = {}
for author in author_counts.keys():
    counter = author_counts[author]
    count = np.array([counter[v] for v in vocabulary])
    freq = (count / np.sum(count)) * 100 # <- percentage
    author_freqs[author] = freq

corpus_freqs = np.mean(list(author_freqs.values()), axis=0)
corpus_stds = np.std(list(author_freqs.values()), axis=0)

## Get vocabulary indicies for most frequent words
# working_index = np.argsort(corpus_freqs)[::-1][:150]
## Can choose diffrent numbers
# incorrectly identifies the author with this amount
#working_index = np.argsort(corpus_freqs)[::-1][:13]
# need at least 14 for "The Monk"
# working_index = np.argsort(corpus_freqs)[::-1][:14]
# Need only two for "The castle of Otranto"
working_index = np.argsort(corpus_freqs)[::-1][:300]
working_means = corpus_freqs[working_index]
working_stds = corpus_stds[working_index]

## Burrows's delta score
def delta(author_freqs, text_counter):
    ## Calculate frequencies for text
    text_count = np.array([text_counter[vocabulary[i]] for i in working_index])
    text_fingerprint = (text_count / sum(text_counter.values())) * 100 # <- percentage

    author_fingerprint = author_freqs[working_index]

    ## Calculate zscores
    text_zscores = (text_fingerprint - working_means) / working_stds
    author_zscores = (author_fingerprint - working_means) / working_stds

    ## Calculate delta
    delta = np.mean(np.absolute(text_zscores - author_zscores))

    return delta

for author in author_freqs.keys():
    score = delta(author_freqs[author], novel_counts[query_filename])

    print("{:.2f} {}".format(score, author))
