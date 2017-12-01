"""
In last week's experiment, we wanted to know how many cups of tea someone would
have to get right for us to be convinced that they are not just guessing randomly.
The question was "is there really a difference that Dr. Bristol can taste?"
The intuition was that the more cups tasted, the more convinced we could be.

Today we're going to look at word counts. A question could be "is there really a 
difference between how two authors use the same word?" or "does author A like 
word w more than author B likes that word?" What makes this complicated
is that most words are rare. Even if two authors *like* some word exactly the same,
that doesn't mean they will both *use* the word the same number of times.

In this script we'll be examining a method for finding distinctive words between
two groups of texts: Dunning's g-test. This method tests if two proportions 
(e.g. word frequencies) are significantly different.

For this script we'll be comparing works by Jane Austen and Charles Dickens. 
We'll read the novels from each author's directory and then perform Dunning's 
g-test for each term in the overall vocabulary.

1. Describe the meaning of each cell in the cotingency table for these values:
    Group 1: 10 out of 50, Group 2: 5 out of 50.

                "Correct"  "Incorrect"

                Austen      Dickens  
                _______________________
               |           |           |
Apple          |           |           |
               |           |           | # counts for word type
               |-----------|-----------|
Not Apple      |           |           |
               |           |           |
               |___________|___________|

                lengths per author


IT means that 10 people out of the 50 people total were put in Group 1,
while 5 people out of 50 were put in Group 2 based on some criterion. 

2. Using the contingency_table and dunning_g methods, calculate Dunning
   g-scores for the following proportions:
   (a) 100/120, 30/55
        >>> a = contingency_table(100,120,30,55)
        >>> dunning_g(a)
            15.590803315104118
   (b) 100/120, 10/12
        >>> b = contingency_table(100,120,10,12)
        >>> dunning_g(b)
            0.0
   (c) 45/100, 105/200
        >>> c = contingency_table(45,100,105,200)
        >>> dunning_g(c)
            1.5018819112174953
   (d) 0/5, 10/25.
        >>> d = contingency_table(0,5,10,25)
        >>> dunning_g(d)
            4.5402667472259406

   How do differences in proportions and differences in counts affect the 
   resulting g-scores?

    A tables with the same proportions results in a dunning-g score of 0.
    As the proportions of a table get closer the score goes to zero. 

    Differences in counts scale the g-score. I found this out by making a new 
    table with the same proportions as (c) but reduced.
        >>> e = contingency_table(9,20,21,40)
        >>> dunning_g(e)
            0.30037638224349905
    We can see here that the proportions in (c) were divided by 5. Doing
        >>> dunning_g(e)*5
            1.5018819112174953
    which is the score of (c).  


3. What do the magnitude of a g-score indicate?

    The magnitutde of a g-score indicates the relationship between two variables. The lower the g-score,
    the more they appear in the same proportion in both documents (0 being the lowest with same proportions)
    The higher the g-score, the more the variables in the documents are used in different proportions.


4. Add code to calculate the Dunning g-score for each term in the vocabulary.
   Print the ten most distinctive words for Austen and Dickens using the
   print_extreme method. What do you think of these terms?

    Most distinctive words used by Jane Austen and Charles Dickens:
    [('she', 5965.8210082481874), ('her', 5934.000823891085), ('t', 4036.9820462951666), ('not', 3197.1803037336999), 
    ('be', 2776.8835248512619), ('emma', 2559.6100710639316), ('elizabeth', 2518.3192741553289), ('said', 2344.7737649750761),
     ('catherine', 2320.6351207218122), ('elinor', 2305.4190513416092)]
    

5. Add code to determine the least distinctive term usage by Austen and 
   Dickens. Print the ten least distinctive terms. Are these terms surprising?

Least distinctive terms used by Jane Austen and Charles Dickens:
    [('him--and', 2.044289870284377e-05), ('retaining', 2.044289870284377e-05), 
    ('scarlet', 2.044289870284377e-05), ('attendance', 1.5688830196425307e-05), ('friendly', 1.5554913624704625e-05), ('dressing', 7.8448069240621976e-06), 
    ('fit', 5.9727324657199432e-06), ('hesitated', 8.498041381699295e-09), ('inform', 8.498041381699295e-09), ('letting', 8.498041381699295e-09)]
    I thought 'fit' was a surprising fit here (HA).

6. How would the existing Dunning g-scores change if we removed words from
   our vocabulary? What if instead we added another novel?

   If you remove words from the vocabulary or added another novel, the past g-scores
   are not valid. You'll need to recompute the existing g-scores to reflect the 
   updated documents.
"""
import re, sys, glob
import numpy as np
from collections import Counter
from scipy.stats import entropy

word_pattern = re.compile("\w[\w\-\']*\w|\w")

def get_counts(author):
    counter = Counter()
    for filename in glob.glob("{}/pg*.txt".format(author)):
        with open(filename, encoding="utf-8") as reader:
            for line in reader:
                line_tokens = word_pattern.findall(line.lower())
                counter.update(line_tokens)
    return counter

austen_counts = get_counts("Austen")
dickens_counts = get_counts("Dickens")
total_counts = austen_counts + dickens_counts
vocabulary = sorted(list(total_counts.keys()))

def contingency_table(count1, total1, count2, total2):
    table = np.zeros((2,2))
    table[0] = count1, count2
    table[1] = total1-count1, total2-count2
    return table

def dunning_g(table):
  rows = table.sum(axis=1)
  cols = table.sum(axis=0)
  score = 2 * table.sum() * (entropy(rows) + entropy(cols) - entropy(table.ravel()))
  return score

# [2: Add code here.]
# total amount of words
num = sum(total_counts.values())
act = sum(austen_counts.values())
dct = sum(dickens_counts.values())

gscores = Counter()
for term in vocabulary:
    gscore = None
    # [4: Add code to calculate a term's Dunning g-score]
    ctable = contingency_table(austen_counts[term], act-austen_counts[term],
                               dickens_counts[term], dct-dickens_counts[term])
    gscore = dunning_g(ctable)
    gscores[term] = gscore

# Print the the top k word-score pairs
def print_extreme(words, scores, k, is_top):
  k_pos = None
  if is_top:
    k_pos = np.argsort(scores)[::-1][:k]
  else:
    k_pos = np.argsort(scores)[:k]
  print(k_pos)
  for i, pos in enumerate(k_pos):
    print("{}. {}: {}".format(i+1, words[pos], scores[pos]))

print("Most distinctive words used by Jane Austen and Charles Dickens:")
print(gscores.most_common()[0:10])
# [4: Add code here.]
print("\nLeast distinctive terms used by Jane Austen and Charles Dickens:")
print(gscores.most_common()[-10:])
# [5: Add code here.]
