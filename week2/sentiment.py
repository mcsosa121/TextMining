### The two readings for this week try to model plot as a cycle of positivity
### and negativity. How well can we measure these constructs in novels?

### In this work we'll look at evaluating documents with respect to a fixed
### vocabulary. I've included two sample sentiment lexicons, one by
### Bing Liu and one from Matt Jockers' "syuzhet" package.

### ( For an interactive look, see this page: https://mimno.infosci.cornell.edu/sentiment/ )

## Example usage: python sentiment.py syuzhet.csv txt/oliver.txt

# 1. There's a bug in the code. All the paragraphs are being scored as 0.
#  Fix it, and describe what was happening.

## The bug in the code is that many of the lines have newline characters at the end which is messing up the scoring
## To fix this we added line=line.rstrip() before we split by comma which removes white space and newline characters
## from the end of a string up to the first non newline or whitespace character.

# 2. I'm using the Counter class from the `collections` package instead of a
#  python `dict`. Consult the documentation https://docs.python.org/3/library/collections.html
#  and describe four features that Counter provides that dict does not.

## a. elements() - returns an iterator over the elements with the elements repeating as many times as their count.
##    (Ex. key a with a count of 4 would return in the iterator as 'a, a, a, a' )
## b. most_common() - returns the most common elements of the counter and their count (element, count) from most common to least.
## c. subtract() - Given two iterables or other mappings such as counter, subtract the count of each key in one from the key in the
##    other. Ex (a:5, b:6).subtract(a:3, b:2) = (a:2,b:4)
## d. Counters do not raise a keyerror when searching for a key that is not within the counter. Instead they return a count of 0

# 3. The directory `txt` contains works by Charles Dickens in the correct format:
#  one paragraph per line. Apply the two lexicons to `tale.txt`. Do they work?
#  Do they agree? Provide specific examples.

## The lexicons work for this example. They also generally agree but differ in their results. One way is that
## the bingliu lexicon uses -1 for negative and +1 for positive while syuzhet can range from [-1,1] for any word.
## Thus syuzhets results are usually not whole numbers while bingliu's is.
## Additionally while both lexicons rated the same passsages as negative they rated them diffrently.
## Take for example the paragraph starting with "The grindstone had a double handle....". This was both rated as one
## of the most negative paragraphs (-15.750000000000002 by syuzhet and -21.0 by bingliu), however, bingliu had this as
## the most negative paragraph, while syuzhet has it as its second most negative passage. Syuzhet has the paragraph
## starting with "In England, there was scarcely an amount of order and protection ..." as its most negative, while
## bingliu did not even rate this in the top ten most negative at all.

# 4. The code is currently just adding up all the scores for each word token.
#  This favors longer documents: if we just repeat the contents twice, the score doubles.
#  What happens if we normalize by document length? In the `score_counts` function,
#  divide the score by the total number of tokens.

## The output has changed from paragraphs to smaller sentances. So larger paragraphs which are larger and may
## be more negative overall end up being smaller than negative sentances because they have more tokens.
## However, looking at both the results from bingliu and syuzhet it seems that they now share more similar
## top ten most negative/positive results.
## "Lost, utterly lost!", "The worst.", "Yes, unhappily.", and "Business seems bad?" all show up in both
## for most negative.
## I would say normalization is a good idea if you want to look at shorter passages as opposed to paragraphs overall.

# 5. Working with your table, create a lexicon for one of the emotions listed on
#  on the board, or choose your own. You may collect additional documents to test
#  your lexicon, please include these if so.

## The word our group chose was "Sad". The hard about this process was thinking about how to create a rating scale. We pondered
## which words are more negative/positive than others. Is "pessimistic" more sad than "melancholy"?
## What happens when emoji's are brought into the mix. To get around this we followed
## the bingliu model which just assigned positively correlated words with 1 and negatively correlated words with -1.
## Overall I thought it was a pretty fun experience to think of all the synomyms we knew for "Sad" with my group.
## It really helped me see how much work bing liu and syuzhet must have put into carefully crafting their lexicons
## so that it would accurately evaluate a given text.
## Although their lexicons are big, they aren't perfect, and I don't believe there is a perfect lexicon for everything.
## However, I think our lexicon did a good job. Testing it out on tale.txt we also found similar top ten results such as
## "Business seems bad?" and "Lost, utterly lost"

# 6. I've set this up so that we are looking at the most extreme passages in
#  the sources. What does this approach show, and what does it hide? How does it
#  affect how we evaluate the tool?

## It shows how lexicons combined wiht textmining can evaluate a passage and find
## the most negative or positive passages in a text based on the rules we define in our lexicon.
## However, this approach hides some of the more moderate passages of the text and so we can't
## really see all of what is being evaluated. This affects how we evaluate the tool
## because we are only seeing these certain results rather than everything given to us.

#### IDEAS FOR YOUR WRITING RESPONSE FOR FRIDAY:

## Both groups are working from Vonnegut's description of plot. Does this
##  view really reflect plot? If not, what is missing, and how important is it
##  to you?
##
## Given your experience with lexicon-based sentiment analysis, how well does
##  it approximate a quantity that's relevant for plot analysis?
##
## Would a more nuanced view of emotion lead to a better representation of plot?


import re, sys
from collections import Counter

## The script takes two command line arguments: the lexicon file and the text file

## Format: each line is [word],[weight]
lexicon_file = sys.argv[1]
## Format: each line is one paragraph
text_file = sys.argv[2]

## Create a mapping from words to numbers
word_weights = {}
with open(lexicon_file) as lexicon_reader:
    for line in lexicon_reader:
        line = line.rstrip()
        weight, word = line.split(",") ## split on comma
        word_weights[word] = float(weight) ## convert string to number

## Here's an example of a simple pattern defining a word token.
word_pattern = re.compile("\w[\w\-\']*\w|\w") ## what matches this?

## Now look at the actual documents. We'll create a list with one object per text segment.
paragraphs = []

## This function applies the word weights to a list of word counts
def score_counts(counter):
    ## accumulate word weights in this variable
    score = 0

    ## count the words in the passage
    total_tokens = sum(counter.values())
    ## check for empty segments
    if total_tokens == 0:
        return 0

    ## for each word, look up its score
    for word in counter.keys():
        if word in word_weights:
            score += word_weights[word] * counter[word]
    return score / total_tokens

## here's where we actually read the file
with open(text_file, encoding="utf-8") as file:

    ## This block reads a file line by line.
    for line in file:
        line = line.rstrip()

        tokens = word_pattern.findall(line)

        ## turn a list into a word->count map
        paragraph_counts = Counter(tokens)

        ## create the paragraph object, with the original text,
        ##  the word counts, and the total score.
        paragraphs.append({ 'text': line, 'counts': paragraph_counts, 'score': score_counts(paragraph_counts) })

## Now sort the objects, in place, by score
paragraphs.sort(key = lambda x: x['score'])

## Display the 10 most negative
for paragraph in paragraphs[0:9]:
    print("{}\t{}".format(paragraph['score'], paragraph['text']))

## ... and the 10 most positive
for paragraph in paragraphs[-10:-1]:
    print("{}\t{}".format(paragraph['score'], paragraph['text']))
