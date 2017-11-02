"""

Topic models look for groups of words that occur frequenty together. We can often recognize these clusters as specific themes that appear in the collection -- thus the term "topic" model.

Our example corpus today is a collection of Viking sagas. Start python like this:

    python -i topicmodel.py sagas_en.txt 20

We will work at the python prompt ">>>".

Today we'll be working with the simplest and most reliable topic model algorithm, Gibbs sampling.
Gibb sampling is a way to take a very complicted optimization problem and break it into little problems that are individually easy.

First, we need to have a way of describing probability distributions.
A discrete distribution is a vector of numbers that are >= 0.0 and sum to 1.0.
One function is called *entropy*. Entropy takes a distribution and returns a number.

1. Run `entropy(np.array([0.7, 0.1, 0.2]))`. What is the value?

The value is 1.1567796494470395.

2. Run `entropy(np.array([7, 1, 2]))`. Does the value change? Why or why not?

The value doesn't change. This is because it is based off the distribution of the numbers rather than their values.

3. Try different (non-negative) values of the three numbers. What is the largest value you can get, and what is the smallest?

The largest value possible is when the probabilities are the same
i.e 
entropy(np.array([1/3,1/3,1/3]))
1.5849625007211561
We determined in class that the largest possible value is log_2(k) where k is the number of elements in the array. 
The smallest possible value is 0. This is achieved when the probabilities are as diffrent as possible.
For example 
entropy(np.array([99999999/100000000,0.000000001,0]))
3.1340048109443602e-08
entropy(np.array([1,0,0]))
0.0
 entropy(np.array([0,1,0]))
0.0

4. Now try different (non-negative) values of *four* numbers. Can you get a larger or smaller entropy than with three?

You can get a larger entropy that with 3 because log_2(3)<log_2(4). 

5. Describe in your own words what entropy is measuring.

Entropy is measuring the unceratinty in a distrobution.Essentially how similar the diffrent values are within the distribution
determines the entropy, with similar giving higher values, and less similar giving lower.

The Gibbs sampling algorithm proceeds in multiple iterations. In each iteration, 
we look at all the word tokens in all the documents, one after another.
For each word, we erase its current topic assignment and sample a new topic 
assignment given all the other word tokens' topic assignments.

Now look at the lines below the "SAMPLING DISTRIBUTION" comment. These define two vectors:
* The probability of each topic in the current document
* The probability of the current word in each topic

We'll look at a particular dramatic moment in Njal's saga. Define these variables:

    document = documents[1160]
    doc_topic_counts = document["topic_counts"]
    word = "sword"
    word_topic_counts = word_topics[word]

Use this command to suppress scientific notation:

    np.set_printoptions(suppress=True)

6. Calculate the entropy of `doc_topic_counts`
 entropy(doc_topic_counts)
 3.0850551027564772

7. Calculate the entropy of `(doc_topic_counts + doc_smoothing)`. Should this be larger or smaller than the previous value?
entropy(doc_topic_counts+doc_smoothing)
3.9971924036969129
This is a little larger than the previous value, because the smoothing element is bringing the distribution towards something more 
smooth (as opposed to chunky).

8. Calculate the entropy of `(word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)`

>>> ww = (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)
>>> ww
array([ 0.00171017,  0.00226594,  0.00188836,  0.0013621 ,  0.00250146,
        0.00204928,  0.00255194,  0.00231713,  0.00259321,  0.00236257,
        0.00249626,  0.0026983 ,  0.00213855,  0.00183114,  0.00204635,
        0.00165762,  0.00161891,  0.00234189,  0.00260293,  0.00274872])
>>> entropy(ww)
4.2977806628527109

9. Calculate the entropy of `(doc_topic_counts + doc_smoothing) * (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)`

>>> ww = (doc_topic_counts + doc_smoothing) * (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)
>>> ww
array([ 0.00085508,  0.00339891,  0.00472089,  0.00068105,  0.00625365,
        0.00307392,  0.0038279 ,  0.00115856,  0.00129661,  0.00118128,
        0.00124813,  0.00404745,  0.00320783,  0.00091557,  0.00102317,
        0.00414406,  0.00080945,  0.00585472,  0.00130147,  0.00137436])
>>> entropy(ww)
3.9845722229198097

These values are random initializations. Let's run the algorithm
over the documents a few times and see what happens. Run:

    sample(25)

Use `print_all_topics()` to get a view of the current state of the topics.

10. This function prints the number of tokens in each topic for the sample doc. Describe how (if at all) they change.
>>> print_all_topics()
king any ship bade might where daughter things father know hand has together our little fell thought house day say
king ship has little things long very himself thought day most may might daughter any winter bade father money tell
king thought know himself together hand has any ship day more bade our things most house money could against named
king day himself bade any most things ship more house fell thought know together against very think where own might
king between most ship where fell more house together yet things day bade might could long while hand has though
king ship has together against fell your bade where thought any know between till more may could day matter yet
king bade day get thought father himself more very daughter give any between summer winter your little like matter named
king ship father together more your has most himself fell things any may long might little day very get against
king against give ship very your hand could day father fell where done little together may himself winter house has
king ship has any long fell himself together things most might wife bade your house little where against winter very
king more ship bade could fell house against may might mind where father yet most tell hand little day brought
king day very could ship against little thought any house fell more tell may himself own our things long ready
king things day himself ride house lay may give together against might get more has first also thought know say
king fell thought any bade may against very ship where give house never most day himself hand together our father
king give ship more himself thought things daughter house fell against winter could has where stood while wife little bade
king any fell more thought may together against father give little ship long things your very bade most winter has
king house ship long little thought winter day more fell himself bade ready very mind give has together wife know
king ship where little know most against very day summer done things might himself never hand winter your bade fell
ship king any house himself little father things winter matter bade against though has very done most day could heard
king himself ship house against more has bade might very any day thought give father little together our get matter

>>> sample(25)
>>> print_all_topics()
head against sat through fell wound wounded hall upon cast wounds under dead feet fire sprang bade forward arm himself
ride slaying together west paid north friends until has hands thither matters done because case goods half put east shame
old himself soon house too another thought any again bear winters while slain never end slay life has comes find
ship sea boat wood sailed south east till wind along north island lay knew weather summer ready company sail could
has your ride day journey any meet thought horses welcome most our without sent company stay brothers place talk hand
ship night ships could bade between side ready through stood day little earl once heard near river fell evening large
money own give matter help atonement your between our pay know silver thought kinsmen make think settled worth award long
winter summer where fared bade far spring thither south goods sent themselves together errand homestead give stayed whole being cattle
strong long our try nor thine fight day better wouldst art against look showed none earth last stand faith like
deemed might himself withal together day horse thought little bade whereas somewhat long laid ever either night days sheep ring
abroad like same thus part offer parted hands each better three matters just stone own gone going young deeds done
wilt shalt art yet hast need answers things deem thyself tell may wise know hath more ever asks stead little
sword hand fell shield ran smote axe once spear turned hard blow thrust fight leapt head cut caught door song
folk things might nought because brought forth even heard therewith thereof peace ever exceeding life lay knew tale befell little
morning house work again lay where bed found like horses along clothes through seen last weapons early upon till blood
may nor get hast neither ill word heart done never things fare yet alone fared though else ways deemed kin
king ready also taken journey wish whom received honour winter rather better people your friendship counsel ship property while wealthy
know think say tell too any woman matter still match spoke talk more most best though women mind get nothing
daughter wife name father named dwelt brother mother very most house lived winter feast brothers sister strong abode children father's
suit against law witness give court say cause words right has notice brought summoned lawful make priest defence case hear

After sampling, the topics seem to change a lot. Before the topics almost all contained "king" but afterwards, only one does. 

11. Recalculate the four entropies we calculated above for the sampling distribution. How are they different?

>>> entropy(doc_topic_counts)
2.353533948577482
>>> entropy(doc_topic_counts+doc_smoothing)
3.7388455119640187
>>> ww = (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)
>>> entropy(ww)
0.61375513123183523
>>> ww = (doc_topic_counts + doc_smoothing) * (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)
>>> entropy(ww)
1.0004972691372933

After sampling the entropy of all of these values has gone done. 

12. What is the value of `word_smoothing`? Previously we added 1.0 in this situation. Why are we using a different value now? Use the concept of entropy in your answer.

The value of 'word_smoothing' here is 0.01. After testing we saw that the entropy values went down. So in this case we were using too big of a smoothing 
parameter. The value of smoothing should be found through testing. 

13. What are Norse sagas about, from the perspective of the model?

From the perspective of this model Norse Sagas are about kings, journeys, ships, sailing, helps, weapons, feasts, and more!

14. I'm removing a list of frequent words, words that are too short, and
words whose first letter is capitalized. Why does removing capitalized words
help? What happens if you remove that check? Is this a good idea?

Removing capitalized words helps because capitalization techniques are diffrent across langueages. Additionally, capitalized words such 
as names may not be as important to analysis. If we remove that check, its not necessarily a good idea because we are removing that 
standardization. Words that are capitalized in Norse sagas may throw off the system.

"""

import re, sys, random, math
import numpy as np
from collections import Counter

word_pattern = re.compile("\w[\w\-\']*\w|\w")

if len(sys.argv) != 3:
    print("Usage: topicmodel.py [docs file] [num topics]")
    sys.exit()

num_topics = int(sys.argv[2])
doc_smoothing = 0.5
word_smoothing = 0.01

stoplist = set()
with open("stoplist.txt", encoding="utf-8") as stop_reader:
    for line in stop_reader:
        line = line.rstrip()
        stoplist.add(line)

word_counts = Counter()

documents = []
word_topics = {}
topic_totals = np.zeros(num_topics)

for line in open(sys.argv[1], encoding="utf-8"):
    #line = line.lower()
    
    tokens = word_pattern.findall(line)
    
    ## remove stopwords, short words, and upper-cased words
    tokens = [w for w in tokens if not w in stoplist and len(w) >= 3 and not w[0].isupper()]
    word_counts.update(tokens)
    
    doc_topic_counts = np.zeros(num_topics)
    token_topics = []
    
    for w in tokens:
        
        ## Generate a topic randomly
        topic = random.randrange(num_topics)
        token_topics.append({ "word": w, "topic": topic })
        
        ## If we haven't seen this word before, initialize it
        if not w in word_topics:
            word_topics[w] = np.zeros(num_topics)
        
        ## Update counts: 
        word_topics[w][topic] += 1
        topic_totals[topic] += 1
        doc_topic_counts[topic] += 1
    
    documents.append({ "original": line, "token_topics": token_topics, "topic_counts": doc_topic_counts })

## Now that we're done reading from disk, we can count the total
##  number of words.
vocabulary = list(word_counts.keys())
vocabulary_size = len(vocabulary)

smoothing_times_vocab_size = word_smoothing * vocabulary_size

def sample(num_iterations):
    for iteration in range(num_iterations):
        
        print(documents[1160]["topic_counts"])
        
        for document in documents:
            
            doc_topic_counts = document["topic_counts"]
            token_topics = document["token_topics"]
            doc_length = len(token_topics)
            for token_topic in token_topics:
                
                w = token_topic["word"]
                old_topic = token_topic["topic"]
                word_topic_counts = word_topics[w]
                
                ## erase the effect of this token
                word_topic_counts[old_topic] -= 1
                topic_totals[old_topic] -= 1
                doc_topic_counts[old_topic] -= 1
                
                ###
                ### SAMPLING DISTRIBUTION
                ###
                
                ## Does this topic occur often in the document?
                topic_probs = (doc_topic_counts + doc_smoothing) / (doc_length + num_topics * doc_smoothing)
                ## Does this word occur often in the topic?
                topic_probs *= (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)
                
                ## sample from an array that doesn't sum to 1.0
                sample = random.uniform(0, np.sum(topic_probs))
                
                new_topic = 0
                while sample > topic_probs[new_topic]:
                    sample -= topic_probs[new_topic]
                    new_topic += 1
                
                ## add back in the effect of this token
                word_topic_counts[new_topic] += 1
                topic_totals[new_topic] += 1
                doc_topic_counts[new_topic] += 1
                
                token_topic["topic"] = new_topic

def entropy(p):
    ## make sure the vector is a valid probability distribution
    p = p / np.sum(p)
    
    result = 0.0
    for x in p:
        if x > 0.0:
            result += -x * math.log2(x)
            
    return result

def print_topic(topic):
    sorted_words = sorted(vocabulary, key=lambda w: word_topics[w][topic], reverse=True)
    
    for i in range(20):
        w = sorted_words[i]
        print("{}\t{}".format(word_topics[w][topic], w))

def print_all_topics():
    for topic in range(num_topics):
        sorted_words = sorted(vocabulary, key=lambda w: word_topics[w][topic], reverse=True)
        print(" ".join(sorted_words[:20]))


