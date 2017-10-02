## The TXT directory contains Shakespeare plays
## from http://lexically.net/wordsmith/support/shakespeare.html
## For each play there's the complete play and then a separate set of files
##  that contain all the lines for each character separately. We'll focus
##  on the "whole play" files.

## 1. What encoding are the files in? Change the encoding
## from UTF-8 to the correct value.

## The correct encoding is utf-16


## 2. What is the format of the files? How can you extract just text? Modify the code
##  to ignore headers, stage directions, and speaker names.

## The format for the files have headers, stage directions, and speaker names
## indside of opening and closing < > brackets. The text of the play is then
## indented once inside these brackets. We can modify the code to ignore these lines
## by checking if the line starts with a tab, else continue.

## 3. Right now we're not saving anything as we read through each play. What data
##  structure should we design so that we can calculate the probability of words
##  by genre?

## I set up another dictionary with counters for each genre which are the
## collectively updated, each time a play is finished.


## 4. Calculate the probability of words in a play and in a genre. For each word
##  in each play, print the word, the two probabilities, and the name of the play
##  and the genre to the screen, separated by tabs, one word per line.

##  Its a lot of output.

## 5. Practice using the command line tools "sort", "head", "tail", and "grep" to
##  organize and view the output of this script.

## The practice was fun! I was able to sort by column with sort, get first
## and last X number of lines with head and tail respectively,
## and use grep to find stuff. Something cool I was able to do was
## first grep to find Hamlet, and then sort by column. Using this,
## I was able to easily solve the Sporcle "Can you name the 100 most
## "Common words in Shakespeare's Hamlet" quiz.

import re, sys, glob
from collections import Counter

genre_directories = { "tragedy" : "TXT/tragedies","comedy" : "TXT/comedies", "history" : "TXT/historical" }

## Here's an example of a simple pattern defining a word token.
word_pattern = re.compile("\w[\w\-\']*\w|\w")
## exclude words that are headers, stage directions, and speaker names
nonword_pattern = re.compile("<[\S ]*>")

# total counter for genres
genre_counter_total = {}
genre_counts = {}
# first loop to populate the data
for genre in genre_directories.keys():
    genre_counter_total[genre] = Counter()
    genre_counts[genre] = {}

    for filename in glob.glob("{}/*.txt".format(genre_directories[genre])):
        play_counter = Counter()
        genre_counts[genre][filename] = play_counter

        with open(filename, encoding="utf-16") as file:  # # What encoding?

            ## This block reads a file line by line.
            for line in file:
                line = line.rstrip()
                ## [Add a rule to ignore non-spoken text here]
                if not line.startswith("\t"):
                    continue

                line = line.lower()

                tokens = word_pattern.findall(line)

                play_counter.update(tokens)
        genre_counter_total[genre].update(play_counter)
        ## Simplest possible output
        # print(play_counter.most_common(20))
    #print(genre_counts[genre][filename].most_common(20))

## second loop to print
print("Word\tP1\tP2\tPlay\tGenre")
for genre in genre_counts.keys():
    for filen in genre_counts[genre].keys():
        for word in genre_counts[genre][filen]:
            # probability within play
            P1 = genre_counts[genre][filen][word] / sum(genre_counts[genre][filen].values())
            # probability within genre
            P2 = genre_counts[genre][filen][word] / sum(genre_counter_total[genre].values())
            print("{}\t{}\t{}\t{}\t{}".format(word,P1,P2,filen,genre))
