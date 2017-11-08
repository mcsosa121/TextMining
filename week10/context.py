"""

What does the context of a word tell us? This will be the basis of word embeddings, but it's good to get some intuitions about what these look like.

We'll use a collection of ghost stories from Gutenberg as an example.

1. Look at the context of words with the `contexts` function. Find three interesting examples with context size 5. (Copy enough output to get a sense of what's there, don't copy everything if it's long.)

- The first interesting example I found was 
    "Spook"
    >>> contexts("spook",5)
    happening at all Poor old [spook] I suppose it would keep
    it There's nothing of the [spook] about Leon He's of this
    of various types The old-fashioned [spook] gradually declines in popularity He
    of the phantom ship the [spook] of the high seas But
    ------------------------------------------------------------------------------
    Interestingly enough, "Spooky" didn't return anything! Spooky!
- The next interesting example, getting in the halloween spirit was
    "Ghost"
    An excert of the output can be seen below
    >>> contexts("ghost",5)
    ...
    a cloud of smoke Gawaine's [ghost] with those of the knights
    romance of _Sir Amadas_ the [ghost] of a merchant whose corpse
    in 1716-17 by a persevering [ghost] called Old Jeffrey whose exploits
    believe it The Cock Lane [ghost] gained very general credit and
    The appearance of Lord Lyttleton's [ghost] in 1779 was described by
    believe in the Cock Lane [ghost] as the most extraordinary thing
    if it was really a [ghost] it could do one no
    story in _Isabella_ of Lorenzo's [ghost] who Moaned a ghostly undersong
    ...abs
- The last interesting example I found was 
    "Petticoat"
    >>> contexts("petticoat",5)
    a glimpse of her old [petticoat] the thought passed as quickly
    are tagging after some new [petticoat] again And he continued wrathfully
    upper stone lay a white [petticoat] on the second a silk
    stone was discovered a white [petticoat] on the second a silk
    upper stone lay a white [petticoat] on the second a silk
    been scarcely possible that the [petticoat] and scarf should have retained
    bushes were broken but the [petticoat] and the scarf are found
    a new slip from the [petticoat] He tore it made it
    when scarcely out of the [petticoat] stage displayed the regular Whittingen
    of every man when a [petticoat] is more attractive to him
    with tarnished lace a satin [petticoat] satin shoes and no stockings

2. Try smaller and larger context windows. What can you get from the larger windows that you can't from the smaller windows? What can you still see from small windows (1 or 2 words)? [hint: try different parts of speech]

From larger windows, you can get a better overall idea, of what part of the story your word occured in, but you lose sight of the context of the words around the specified word. From only 1 or 2 word windows
you can see how people used the words. For example with a 2 word window you can see "Poor old [spook] I suppose" which probably meant that it was referring to a person or thing. This is a lot easier
to see in these small windows.

3. Now use `nearby_words` to summarize these contexts. Use the Counter `most_common()` function if the results are too large. How much can you tell about a word from its top 20 most frequently co-occurring words? How does the window size affect this?

>>> nearby_words("spook",20)
Counter({'of': 9, 'in': 6, 'the': 6, 'was': 5, 'I': 4, 'it': 4, 'that': 4, 'at': 2, 
         'nothing': 2, 'this': 2, 'world': 2, 'and': 2, 'The': 2, 'are': 2, 'a': 2, 'with': 2, 'daylight': 1, 
         '_': 1, 'another': 1, 'voice': 1, 'muttering': 1, 'wish': 1, 'sure': 1, 'happening': 1, 'all': 1, 'Poor': 1,
         'old': 1, 'suppose': 1, 'would': 1, 'keep': 1, 'its': 1, 'feet': 1, 'if': 1, 'her': 1, 'deck': 1, 'quite': 1, 
         'vertical': 1, 'Think': 1, "she'll": 1, 'go': 1, 'down': 1, 'or': 1, 'just': 1, 'melt': 1, 'jumps': 1, "can't": 1, 
         'bear': 1, 'to': 1, 'think': 1, 'him': 1, 'night': 1, 'Rubbish': 1, 'Curtis': 1, 'growled': 1, 'You': 1, 'imagine': 1, 
         "There's": 1, 'about': 1, 'Leon': 1, "He's": 1, 'but': 1, 'It': 1, 'odd': 1, 'however': 1, 'from': 1, 'time': 1, 'he': 1, 
         'White_': 1, 'had': 1, 'appeared': 1, 'six': 1, 'years': 1, 'earlier': 1, '_Blackwood_': 1, 'stories': 1, 'included': 1, 'these': 1, 
         'magazines': 1, 'various': 1, 'types': 1, 'old-fashioned': 1, 'gradually': 1, 'declines': 1, 'popularity': 1, 'He': 1, 'is': 1, 'ousted': 1, 
         'scientific': 1, 'age': 1, 'by': 1, 'more': 1, 'recondite': 1, 'forms': 1, 'terror': 1, 'Before': 1, '1875': 1, 'MISFIT': 1, 'GHOST': 1, 
         'Every': 1, 'boy': 1, 'knowledge': 1, 'adventurous': 1, 'literature': 1, 'otherwise': 1, 'novels': 1, 'action': 1, 'knows': 1, 'phantom': 1, 
         'ship': 1, 'high': 1, 'seas': 1, 'But': 1, 'has': 1, 'not': 1, 'been': 1, 'known': 1, 'ships': 1, 'themselves': 1, 'haunted': 1, 'service': 1})

Looking at this output you can kind of tell which kinds of words occur near your selected word. Using this with the context window of the word, you can 
probably get a good idea, of how the word appears within your corpus. 

4. Use the `smoothed_kl` function to measure the divergence between Counters. You can either generate these from the `nearby_contexts` function with window size 2 or use the `total_counts` Counter for the whole collection. Compare five words to the `total_counts` distribution. What word has the least divergence? Why do you think that might be so?

>>> v1 = nearby_words("spook",2)
>>> v2 = nearby_words("ghost",2)
>>> v3 = nearby_words("petticoat",2)
>>> v4 = nearby_words("haunted",2)
>>> v5 = nearby_words("cat",2)
>>> smoothed_kl(v1,total_counts)
3.573971988063461
>>> smoothed_kl(v2,total_counts)
2.00025275171302
>>> smoothed_kl(v3,total_counts)
3.4155417031015047
>>> smoothed_kl(v4,total_counts)
2.1441010600927126
>>> smoothed_kl(v5,total_counts)
1.871153701448988

In this case, Cats has the lowest divergeance. I believe that is because looking at the length of each nearby_words vector, cats has the most. This means it occurs the most compared to the 
other words comparatively, and thus is more similar overal to the total_corpus.

5. Compute the same values, but with three progressively larger window sizes. What happens to the KL divergence? What happens to the relative differences in KL divergence?

>>> v1 = nearby_words("spook",5)
>>> v2 = nearby_words("ghost",5)
>>> v3 = nearby_words("petticoat",5)
>>> v4 = nearby_words("haunted",5)
>>> v5 = nearby_words("cat",5)
>>> smoothed_kl(v1,total_counts)
3.475535460002283
>>> smoothed_kl(v2,total_counts)
1.5576819200368026
>>> smoothed_kl(v3,total_counts)
3.244803484702338
>>> smoothed_kl(v4,total_counts)
1.6440147572290253
>>> smoothed_kl(v5,total_counts)
1.4205902212869566
>>> v1 = nearby_words("spook",10)
>>> v2 = nearby_words("ghost",10)
>>> v3 = nearby_words("petticoat",10)
>>> v4 = nearby_words("haunted",10)
>>> v5 = nearby_words("cat",10)
>>> smoothed_kl(v1,total_counts)
3.332958953089572
>>> smoothed_kl(v2,total_counts)
1.240606796428148
>>> smoothed_kl(v3,total_counts)
2.976664370070254
>>> smoothed_kl(v4,total_counts)
1.323986241949634
>>> smoothed_kl(v5,total_counts)
1.146788006328511
>>> v1 = nearby_words("spook",50)
>>> v2 = nearby_words("ghost",50)
>>> v3 = nearby_words("petticoat",50)
>>> v4 = nearby_words("haunted",50)
>>> v5 = nearby_words("cat",50)
>>> smoothed_kl(v1,total_counts)
2.710767937718538
>>> smoothed_kl(v2,total_counts)
0.8131880200535309
>>> smoothed_kl(v3,total_counts)
2.1205436346217477
>>> smoothed_kl(v4,total_counts)
0.8302656383446398
>>> smoothed_kl(v5,total_counts)
0.8360243291487914

As you get bigger and bigger windows the KL divergeance of each words neighbors goes down. And as you get larger windows, the relative difference between each example 
goes down as well. With a window size of 50, "haunted" which traditionally had a larger KL divergeance than "cat" now had a smaller KL divergeance!!!!!! 

6. Show five examples of pairs of words that you think are similar and five that you think are not. Show multiple window sizes for each pair. Do these results match your intuitions?

A. 'Haunted' and 'House' 
        >>> w1 = nearby_words("haunted",2)
        >>> w2 = nearby_words("house",2)
        >>> smoothed_kl(w1,w2)
        2.384062497429164
        >>> w1
        Counter({'the': 59, 'by': 42, 'a': 37, 'was': 31, 'in': 29, 'and': 27, 'of': 26, 'house': 25,....})
        >>> w2
        Counter({'the': 1652, ... 'haunted': 25, .....})
        >>> w1 = nearby_words("haunted",10)
        >>> w2 = nearby_words("house",10)
        >>> smoothed_kl(w1,w2)
        1.764069839596492
        >>> w2 = nearby_words("house",50)
        >>> w1 = nearby_words("haunted",50)
        >>> smoothed_kl(w1,w2)
        1.130659839596492
    You can see that the words appear highly in eachothers counters. And as the window size goes up their KL divergeance goes down!!!!
B. "Graveyard" and "Night"
        >>> w1 = nearby_words("graveyard",2)
        >>> w2 = nearby_words("night",2)
        >>> smoothed_kl(w1,w2)
        3.369956297489151
        >>> w2 = nearby_words("night",10)
        >>> w1 = nearby_words("graveyard",10)
        >>> smoothed_kl(w1,w2)
        4.336952709676273
        >>> w1 = nearby_words("graveyard",50)
        >>> w2 = nearby_words("night",50)
        >>> smoothed_kl(w1,w2)
        3.5836375563840286
    I would have thought that this pair would have been close. But they are like Brad and Angelina and are NOT!
C. "Soup" and "Food"
        >>> w1 = nearby_words("soup",2)
        >>> w2 = nearby_words("food",2)
        >>> smoothed_kl(w1,w2)
        0.8818864971454097
        >>> w1 = nearby_words("soup",10)
        >>> w2 = nearby_words("food",10)
        >>> smoothed_kl(w1,w2)
        2.003229425333867
        >>> w1 = nearby_words("soup",50)
        >>> w2 = nearby_words("food",50)
        >>> smoothed_kl(w1,w2)
        2.4712752259255173
    In this, when looking at small windows, soup and food are very similar (one could even say that Soup is a food...),
    but when looking at learger windows it seems like the areas these words appear in are not as similar. 
D. "Black" and "Cat"
        >>> w1 = nearby_words("black",2)
        >>> w2 = nearby_words("cat",2)
        >>> smoothed_kl(w1,w2)
        2.3806246473116786
        >>> w1 = nearby_words("black",10)
        >>> w2 = nearby_words("cat",10)
        >>> smoothed_kl(w1,w2)
        1.8084109957056251
        >>> w1 = nearby_words("black",50)
        >>> w2 = nearby_words("cat",50)
        >>> smoothed_kl(w1,w2)
        1.2084221858203739
    Seemed to be generally similar.
E. "Death" and "Life"
        >>> w1 = nearby_words("death",2)
        >>> w2 = nearby_words("life",2)
        >>> smoothed_kl(w1,w2)
        1.7168592077657634
        >>> w1 = nearby_words("death",10)
        >>> w2 = nearby_words("life",10)
        >>> smoothed_kl(w1,w2)
        1.1299026326450607
        >>> w1 = nearby_words("death",50)
        >>> w2 = nearby_words("life",50)
        >>> smoothed_kl(w1,w2)
        0.568299896096338
    I expected these words to be similar and I was right!
"""

import sys, glob, re, math
from collections import Counter

word_pattern = re.compile("\w[\w\-\']*\w|\w")

## For simplicity, we'll just load the entire collection into one big list of tokens, ignoring boundaries between documents.
tokens = []

for filename in glob.glob("pg*.txt"):
    with open(filename, encoding="utf-8") as reader:
        for line in reader:
            line_tokens = word_pattern.findall(line)
            
            tokens.extend(line_tokens)

total_tokens = len(tokens)
total_counts = Counter(tokens)
vocabulary_size = len(total_counts.keys())

### The keyword-in-context view (KWIC)
def contexts(word, n):
    for i in range(total_tokens):
        if tokens[i] == word:
            ## Show the n previous and n subsequent words
            pre_context = " ".join(tokens[i-n:i])
            post_context = " ".join(tokens[i+1:i+n+1])
            print("{} [{}] {}".format(pre_context, tokens[i], post_context))

### A summary of the words that appear near a target word
def nearby_words(word, n):
    counter = Counter()
    for i in range(total_tokens):
        if tokens[i] == word:
            ## Count up the n previous and n subsequent words
            counter.update(tokens[i-n:i])
            counter.update(tokens[i+1:i+n+1])
    return counter

### Treat two Counters as probability distributions, and calculate their divergence
def smoothed_kl(p, q):
    value = 0.0
    
    sum_p = sum(p.values())
    sum_q = sum(q.values())
    
    for word in total_counts.keys():
        p_prob = (p[word] + 0.01) / (sum_p + 0.01 * vocabulary_size)
        q_prob = (q[word] + 0.01) / (sum_q + 0.01 * vocabulary_size)
        value += p_prob * math.log2(p_prob / q_prob)
    return value
