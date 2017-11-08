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

>>> v1 = get_vector("dispersion")
>>> v2 = get_vector("rhythmic")
>>> v3 = get_vector("calculus")
>>> v11 = get_vector("steadiness")
>>> v22 = get_vector("repellent")
>>> v33 = get_vector("repairer")

>>> nearest(v1+v11, 20)
[(0.98607891518231849, 'dispersion'), (0.98607891518231827, 'steadiness'), (0.95209554381113026, 'inconveniences'), (0.94477513945281932, 'trespassers'), 
 (0.94120615471444136, 'sequestered'), (0.94114937903905582, 'ineffective'), (0.93980199585869961, '_jane'), (0.93965548095856133, 'shifts'), 
 (0.93926265332512227, 'automata'), (0.9371021541843525, 'two-thirds'), (0.93683668399846121, 'disorderly'), (0.93638106027648438, 'maltese'), (0.9355696030597479, 'booty'), 
 (0.93521048045782629, 'renting'), (0.93474258614473382, 'paraphernalia'), (0.93473779947972058, 'personalities'), (0.93420943948421076, 'pretentious'), (0.93394382879986504, 'rustics'),
 (0.93223605336363868, 'approbation'), (0.93168101085982091, 'bestowing')]

>>> nearest(v2+v22, 20)
[(0.98633723908224002, 'rhythmic'), (0.98633723908224002, 'repellent'), (0.95922873725065361, 'candlelight'), (0.95814919049626734, 'mistook'), 
 (0.95623207378032937, 'embraces'), (0.95398769837685604, 'metaphor'), (0.95008623065075748, 'medicinal'), (0.95004733260326002, 'roughness'), (0.94966754929766584, 'lingers'),
 (0.94833360591694049, 'undecipherable'), (0.9478137519644223, 'merging'), (0.94577422438496195, 'gloss'), (0.94436086373034556, 'strands'), (0.94425093715156372, 'turgid'), 
 (0.94399475556653833, 'manageable'), (0.94389725366955635, 'transparency'), (0.94376530654020718, 'banishing'), (0.94353854999053177, 'disagreeably'), (0.94290478434896574, 'symmetry'), 
 (0.94124356477165139, 'child-like')]

>>> nearest(v3+v33, 20)
[(0.97843649322524073, 'repairer'), (0.97843649322524073, 'calculus'), (0.94286672417739525, 'reputations'), (0.91544436385717942, 'berlin'), (0.91327555343087052, "anything'"),
 (0.91032048681341693, 'plagues'), (0.90988343350161949, 'necessities'), (0.90959311462454984, 'inter'), (0.90852475192773796, 'newton'), (0.90598118391568905, 'disciple'), 
 (0.90570391348114865, '179'), (0.90565547159064375, 'validity'), (0.90498246950448258, 'prophets'), (0.9046492257089005, 'incensed'), (0.90453653082941909, 'algebraical'), 
 (0.90414686875560712, 'bono'), (0.90333328589978423, 'locke'), (0.90321458259473375, 'indolent'), (0.90280603340730892, 'epigram'), (0.90239666832878274, '(strange')]

Here the single closest verctor is one of teh two words. This is diffrent than the previous problem in that in the previous example, the most similar vector 
was the word being inputted with a cosine similarity of 1, but here, no word is exactly the same 


5. What about subtraction? Use the same examples as in the previous problem, 
but this time search for the *difference* between vectors (use `-`). 
Comment on what changes and what stays the same.

>>> nearest(v1+v11, 20)
[(0.98607891518231849, 'dispersion'), (0.98607891518231827, 'steadiness'), (0.95209554381113026, 'inconveniences'), (0.94477513945281932, 'trespassers'), 
 (0.94120615471444136, 'sequestered'), (0.94114937903905582, 'ineffective'), (0.93980199585869961, '_jane'), (0.93965548095856133, 'shifts'), (0.93926265332512227, 'automata'), 
 (0.9371021541843525, 'two-thirds'), (0.93683668399846121, 'disorderly'), (0.93638106027648438, 'maltese'), (0.9355696030597479, 'booty'), (0.93521048045782629, 'renting'), 
 (0.93474258614473382, 'paraphernalia'), (0.93473779947972058, 'personalities'), (0.93420943948421076, 'pretentious'), (0.93394382879986504, 'rustics'),
 (0.93223605336363868, 'approbation'), (0.93168101085982091, 'bestowing')]
>>> nearest(v2-v22, 20)
[(0.36523887313186021, 'gentle'), (0.2887523250808447, 'lower'), (0.27761981175209077, 'constant'), (0.24033043104632457, 'high'), (0.22691161094833923, 'valleys'), 
 (0.22569432047943083, 'earth-current'), (0.22259040678929715, 'ears'), (0.22233415287260422, 'alway'), (0.21950528145704964, 'seas'), (0.21940837352240242, 'steady'), 
 (0.21692397948624051, 'millions'), (0.2145133018498169, 'forces'), (0.21425651810642329, 'music'), (0.21030658571618779, 'roar'), (0.21022036299572924, 'regular'), 
 (0.2084514131778693, 'south-west'), (0.20776329152482262, 'hearts'), (0.20611505294818902, 'volcanoes'), (0.20519826920825276, 'higher'), (0.20165896119872664, 'sway')]
>>> nearest(v3-v33, 20)
[(0.48247076845169395, 'vapor'), (0.44859851773914872, 'surfaces'), (0.43699120254941859, 'dense'), (0.43492835747660741, 'funnel'), (0.43295040791122685, 'finely'), 
 (0.43200596686560938, 'hues'), (0.43031867326025686, 'resembling'), (0.42944567008517059, 'rim'), (0.41650125098692975, 'thickly'), (0.41602337925518929, 'grotesquely'), 
 (0.41335588962344988, 'surface'), (0.40631706972141124, 'vellum'), (0.40627059764857643, 'portions'), (0.40585150216634702, 'machinery'), (0.40495213011372228, 'species'),
 (0.4047654670046551, 'spots'), (0.40438733726491083, 'diameter'), (0.40333776419301909, 'dusky'), (0.40294993132451229, 'tapestry'), (0.40213682399400646, 'patterns')]

Here the closest word is in 2/3 cases, not one of the inputted words. This shows that the difference brings out different words when looking at words that are similar.

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
