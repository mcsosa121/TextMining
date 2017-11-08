"""

Like LSA and topic modeling, the goal of word embeddings is to create a numeric representation of the meaning of words. 
I've given you three word embedding matrices. (I use "word embedding" and "word vector" interchangeably.)

Start with `horror.vec`, which is an embedding trained on our collection of horror fiction:

    python -i word_vectors.py horror.vec

1. The function `get_vector()` retrieves the vector for a string. Pick a word and get its vector. What is the dimension of this vector?

I chose the word dispersion. 
>>> get_vector("dispersion")
array([ 0.17645026,  0.12920317,  0.00480639,  0.00854916, -0.00315589,
        0.06632598, -0.15313362, -0.08383843,  0.03145169,  0.00142347,
       -0.04867605,  0.25591017, -0.15826508,  0.01413075, -0.06094327,
        0.08058932,  0.00698542,  0.01380502, -0.04847799, -0.12099447,
        0.10903742,  0.13108552, -0.03313353,  0.10541184, -0.13854947,
        0.03019624,  0.09590186,  0.00513384, -0.07726246,  0.01901342,
        0.01111326, -0.04571502, -0.0884461 ,  0.02042682, -0.08902718,
       -0.05494674, -0.09366759, -0.09905276, -0.14295254,  0.05319615,
       -0.15531879,  0.02872145, -0.05775554,  0.13041442,  0.08309367,
       -0.20132189,  0.13005432, -0.00681985, -0.02114375,  0.04288822,
       -0.09942923,  0.0329232 , -0.14183949,  0.05534204, -0.1057474 ,
       -0.18850551, -0.08211157,  0.08806963,  0.13940062,  0.02712718,
        0.06240332, -0.02342385,  0.12499651,  0.12100265,  0.0359186 ,
        0.02226662, -0.01411193,  0.00696889, -0.08752129, -0.27827745,
       -0.07923321, -0.11315405,  0.08486145,  0.02295408, -0.11319497,
        0.07375474,  0.11163998,  0.01125975,  0.17548453,  0.05814429,
        0.06182716, -0.00919407,  0.06434951,  0.13309882,  0.10266197,
        0.03053834,  0.10724509,  0.05815002, -0.13067631, -0.00798143,
        0.02010354, -0.08533613,  0.29081557, -0.10826811, -0.06876895,
       -0.08648191, -0.11666505,  0.11986505, -0.0217371 ,  0.01949628])
The dimension is 100 by 1.

2. The function `nearest()` sorts all words by their cosine similarity to a specified vector. The arguments are a vector and a number of nearest neighbors. 
Search for three or more words and copy the 20 nearest neighbors here. What is the closest word to your query vector, and what is its cosine similarity?

A. "dispersion"
[(1.0, 'dispersion'), (0.94470325393427623, 'steadiness'), (0.94281227407776935, 'ineffective'), (0.93763216813255768, '_jane'), (0.9358813523072459, 'pretentious'), 
 (0.93584879298220192, 'liqueurs'), (0.93524136325385698, 'inconveniences'), (0.93417761256928211, 'disorderly'), (0.93253300245322346, 'automata'), (0.92995770070118189, 'flocked'),
 (0.92965349162116062, 'rustics'), (0.92951327832118935, 'departments'), (0.92929798652175089, 'archipelago'), (0.92929333573498785, 'sequestered'), (0.92790018730510482, 'booty'), 
 (0.92764670645630698, 'condensed'), (0.92727875611112331, 'trespassers'), (0.9264762499674617, 'renting'), (0.92609541840119725, 'stragglers'), (0.92580998311829354, 'pursues')]
 Most similar word 'Steadiness' has a cosine similarity of 0.94.
B. "rhythimic"
[(1.0, 'rhythmic'), (0.94572229840075217, 'repellent'), (0.94424555900057716, 'embraces'), (0.94128740161840485, 'medicinal'), (0.93951936965893879, 'lingers'), 
 (0.93860573372616418, 'inefficient'), (0.93678080705657307, 'metaphor'), (0.93342388489916217, 'mistook'), (0.93269664715539691, 'candlelight'), (0.93191576730332981, 'symmetry'), 
 (0.93162629717569567, 'banishing'), (0.93149101380013732, 'turgid'), (0.93114736178610691, 'merging'), (0.93034162794082231, 'resistless'), (0.92963850725752151, 'roughness'), 
 (0.9295002175055721, 'transparency'), (0.92733355982168852, 'strands'), (0.92674564943723514, 'gloss'), (0.92645735850469801, 'hymns'), (0.92600093862261224, 'unnerving')]
 Most similar word "repellent' has a cosine similarity of 0.9445.
C. "calculus"
[(1.0, 'calculus'), (0.91467594254981266, 'repairer'), (0.90552404388191898, '179'), (0.90536847101722162, 'newton'), (0.90099654872970714, 'cui'), (0.89942774021148997, 'berlin'), 
 (0.89707111021221952, 'inter'), (0.89669957999083816, 'reputations'), (0.89665777938502012, 'indolent'), (0.8957951126055872, 'suppression'), (0.89549849451769481, "spenser's"), 
 (0.89487284421256197, 'bono'), (0.89461409429850569, 'validity'), (0.89420753678002729, 'disciple'), (0.89389559473508939, 'landscape-gardening'), (0.89237604324351749, 'verify'), 
 (0.89171935626066212, 'plagues'), (0.89159867785146085, 'contribution'), (0.89091859185919831, 'priori'),(0.89076537824989432, 'microscope')]
 Most similar word 'repairer' has a cosine similarity of 0.914. 


3. Go back to the `context.py` script, and search for one of the same words, using context-size five. (Use `.most_common(100)` to avoid pages and pages of output.) 
Now do the same for some of the words that the embedding said were nearest. Are their context words similar?

A lot of my words didn't have much context. For example "contexts("rythmic",5)" returned nothing.
"contexts("calculus",5)" returned
a lecherous text-book of the [calculus] or of a reporter's story
but "contexts("repairer",5)" didn't return anything.
We lucked out with 
>>> contexts("dispersion",5)
length to the determination of [dispersion] over the adjacent country in
a loud tone A general [dispersion] of the crowd ensued and
have been expected our complete [dispersion] and the arrest of some
>>> contexts("steadiness",5)
And so hauling with great [steadiness] we brought the monster near
car consequently followed with a [steadiness] so perfect that it would
rather for the quietude and [steadiness] of its population than for
was the suddenness and the [steadiness] of the attack that had
walk with some degree of [steadiness] among temptations and in my
walk with some degree of [steadiness] among temptations and in my

It seems like their context words were a bit similar but not too much. 


4. Go back to the embedding vectors. Now try searching for the sum of two vectors: 
just use `+`! Search for three pairs of words, and copy the 20 nearest neighbors here. 
What do you notice about the single closest vector? How is this different from the previous question?

[Response here]

5. What about subtraction? Use the same examples as in the previous problem, 
but this time search for the *difference* between vectors (use `-`). 
Comment on what changes and what stays the same.

[Response here]

6. The file `original.vec` is trained in the same way on the same files, but without modifications to the text.
   Load these vectors instead of `horror.vec`. Run the same queries. What do you think is different about how I pre-processed the training data?

[Response here]

7. The file `ecco_vectors.vec` has embeddings trained by Ryan Heuser on billions of words from 18th century books. 
   Load this file, and try at least five queries. This can include single words or combinations of words. How are these different 
   from the vector results from the smaller Horror fiction data set? What do you notice about words in 18th century books?

[Response here]

"""

import numpy as np
import sys

reader = open(sys.argv[1], encoding="utf-8")

# The first line of the file gives the dimensions of the matrix
first_line = reader.readline()
fields = first_line.split(" ")
n_rows = int(fields[0])
n_cols = int(fields[1])

word_vectors = np.zeros((n_rows, n_cols))
vocabulary = []
word_ids = {}

for line in reader:
    fields = line.strip().split(" ")
    word = fields[0]
    vector = np.array(fields[1:], dtype=float)
    norm = np.linalg.norm(vector)
    
    # in some odd cases a word may have no vector
    if norm == 0.0:
        continue
    
    # record the word and its row position in the matrix
    word_id = len(vocabulary)
    vocabulary.append(word)
    word_ids[word] = word_id
    
    word_vectors[word_id] = vector / norm

## look up the id for a word, and grab the row associated with that word
def get_vector(word):
    return word_vectors[ word_ids[word] ]

## define a shortcut
v = get_vector

def nearest(v, n):
    norm = np.linalg.norm(v)
    if norm > 0.0:
        v /= norm
    return(sorted(zip(word_vectors.dot(v), vocabulary), reverse=True)[:n])
