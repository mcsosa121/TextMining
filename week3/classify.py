## The TXT directory contains Shakespeare plays from http://lexically.net/wordsmith/support/shakespeare.html
## For each play there's the complete play and then a separate set of files
##  that contain all the lines for each character separately. We'll focus
##  on the "whole play" files.

## 1. The code currently causes a "math domain error". Why, and how can you fix it?

## The code is trying to take the log of 0. This happens when we try to find the probability
## of a word not in the vocabulary, which is 0. To get around this we add the following lines
## genre_counter[word]+1
## This ensures that the probability will never be 0.

## 2. We're adding up the log probabilities of the words. Why is the log function important?
##  What would happen if we calculated the probability of a play instead of the log probability?

## It's because the probabilities of words calculated can sometimes be very small.
## Taking the product of these probabilities makes the total probability even smaller. Like Professor Mimno
## mentioned in class, it only takes arond 30 words before we get underflow.
## If we tried to calculate the probability of an entire play then that would happen.
## To combat this, we take the log probability which ends up being a sum rather than a product.
## However, like we saw in question 1, we must be careful to ensure that the probability of any
## word is not 0, since log is not defined there.


## 3. Look at the log probability of each play according to each genre's language model.
##  What do you think of these numbers? How does the length of the play relate?

## I thought it was pretty interesting to see how the log probability depended on the
## length of each play.
## For longer plays the probability was smaller compared to shorter plays.


## 4. How accurate is the classifier? Are we allowing the classifier to "peek" at the
##  data we're asking it to classify? Modify the code to remove the effect of each play in turn.

## The classifier is not as accurate as we want it to be. It was originally including the
## testing data that we were training on, which is a big sin.


## 5. Reevaluate your classifier. What has changed? What do you think of this difference?

## The classifier is now a bit more accurate, but it still missclassifies a lot of the plays.
## I was suprised at some of the decisions it made, like classifying Romeo and juliet as more
## of a Comedy than Timon of Athens

## 6. Why is the classifier making the decisions it does? What separates "comedy" words from "tragedy"
##  words? How can you pick apart the computation we're doing here to measure the impact of specific words?
##  Work on this part in groups.
## The computation being done here is the sum of all the log probabilities.

import re, sys, glob, math
from collections import Counter

genre_directories = { "tragedy" : "TXT/tragedies", "comedy" : "TXT/comedies", "history" : "TXT/historical" }

word_pattern = re.compile("\w[\w\-\']*\w|\w")

genre_play_counts = {}
genre_counts = {}
all_counts = Counter()

for genre in genre_directories.keys():

    genre_play_counts[genre] = {}
    genre_counts[genre] = Counter()

    for filename in glob.glob("{}/*.txt".format(genre_directories[genre])):

        play_counter = Counter()

        genre_play_counts[genre][filename] = play_counter

        with open(filename, encoding="utf-16") as file: ## What encoding?

            ## This block reads a file line by line.
            for line in file:
                line = line.rstrip()
                if not line.startswith("\t"):
                    continue

                line = line.lower()

                tokens = word_pattern.findall(line)

                play_counter.update(tokens)

        genre_counts[genre] += play_counter
        all_counts += play_counter

vocabulary = all_counts.keys()
vocabulary_size = len(vocabulary)

genres = genre_play_counts.keys()

## Now loop through all plays and decide which genre they match most closely

print("\t".join(genres))

# acutal genre of the play
for real_genre in genres:

    for filename in genre_play_counts[real_genre].keys():
        play_counter = genre_play_counts[real_genre][filename]
        play_sum = sum(play_counter.values())

        genre_scores = {}

        genre_counts[real_genre] -= play_counter

        # loop through all possible genres
        for genre in genres:
            genre_counter = genre_counts[genre]
            genre_sum = sum(genre_counter.values())
            genre_scores[genre] = 0

            for word in play_counter:
                p_word_in_genre = (genre_counter[word]+1) / (genre_sum + vocabulary_size)
                genre_scores[genre] += play_counter[word] * math.log(p_word_in_genre)

            genre_scores[genre] /= play_sum

        max_score = max(genre_scores.values())

        genre_counts[real_genre] += play_counter

        print("{:.3f}\t{:.3f}\t{:.3f}\t{}\t{}".format(genre_scores['tragedy'], genre_scores['comedy'], genre_scores['history'], play_sum, filename))
