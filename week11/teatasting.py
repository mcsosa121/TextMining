"""
Randomization Tests

Load this script interactively:

    python -i teatasting.py

Goal: We often want to make arguments that variables are related. For example, we might want to argue that one author uses a word more often than another, or that 19th C novels use longer sentences than 20th century novels. But with finite and often small samples, we could find patterns that are really just random chance. How do we tell whether two variables are actually related, or if there is only chance similarity?

The key question is: what would I observe if there were no connection between the two variables? An experiment is *statistically* convincing if the pattern I saw is sufficiently unlikely by random chance. But what do "unlikely" and "sufficiently" mean?

We'll start by replicating the "Tea Tasting" experiment from R.A. Fisher's book "The Design of Experiments" (1935). Here the two variables are (1) whether milk was added to a cup before or after the tea and (2) whether a taster says the milk was added before or after. 

1. Run the function "guess_equal()" with n=8, 10 times. Record your results here:
>>> counts = []
>>> for i in range(10):
...   n=guess_equal(eight_cups)
...   counts.append(n)
...
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 1, 0, 0, 0, 0, 1, 1]
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 0, 0, 1, 0, 0, 1, 1]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 1, 0, 1, 1, 0, 1, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 0, 1, 1, 0, 1, 1, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 1, 1, 0, 0, 0, 0, 1]
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 0, 0, 1, 0, 1, 0, 1]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 1, 0, 0, 1, 1, 1, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 1, 0, 1, 0, 1, 1, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 1, 1, 1, 0, 0, 0, 1]
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 1, 1, 0, 0, 0, 1, 0]
>>> counts
[6, 4, 6, 2, 4, 2, 6, 4, 2, 6]
>>> c = Counter(counts)
>>> c
Counter({6: 4, 4: 3, 2: 3})

It seems here that the top results here was 6 and 4. However, I ran this another time and 4 and 2 were the top results. So it can vary.

2. How important is it that we know how many positive examples there are? Run the function "guess_randomly()", also with n=8, 10 times. Record your results. How are these results different from #2? Would you be more or less willing to tolerate a mistake?

>>> counts = []
>>> for i in range(10):
...   n = guess_randomly(eight_cups)
...   counts.append(n)
...
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 0, 1, 0, 1, 0, 1, 1]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 1, 0, 0, 1, 1, 0, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 0, 1, 0, 0, 0, 0, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 0, 0, 0, 0, 1, 0, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 1, 1, 0, 1, 1, 0, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 0, 1, 0, 0, 0, 1, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 0, 0, 0, 0, 0, 1, 1]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 1, 1, 1, 0, 0, 0, 0]
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 0, 1, 1, 0, 1, 0, 1]
[1, 1, 0, 0, 1, 0, 1, 0]
[1, 0, 1, 1, 1, 0, 1, 1]
>>> counts
[5, 5, 3, 3, 4, 5, 5, 3, 0, 4]
>>> c = Counter(counts)
>>> c
Counter({5: 4, 3: 3, 4: 2, 0: 1})

Here we can see that 5 was the top result. 
>>> i = 0
>>> n = guess_randomly(eight_cups)
[1, 1, 0, 0, 1, 0, 1, 0]
[0, 0, 1, 0, 0, 1, 0, 0]
>>> while(n is not 8):
...   n = guess_randomly(eight_cups)
...   i=i+1
Running a script like this we found that we needed 136 trials to get a guess that guessed all 8 cups correctly.

3. In the `run_experiments` function, write a "for" loop that runs `guess_equal` function `num_trials` times. Use the `results` Counter to keep track of how many times you get each number of correct guesses. Run this function 10 times with 1000 trials each, and record your results here:

>>> for i in range(10):
...   run_experiments(eight_cups, 1000)
...
Counter({4: 528, 2: 228, 6: 224, 8: 12, 0: 8})
Counter({4: 533, 6: 222, 2: 221, 8: 15, 0: 9})
Counter({4: 489, 6: 248, 2: 236, 0: 17, 8: 10})
Counter({4: 496, 6: 241, 2: 224, 0: 24, 8: 15})
Counter({4: 506, 6: 239, 2: 223, 0: 16, 8: 16})
Counter({4: 531, 2: 238, 6: 195, 0: 22, 8: 14})
Counter({4: 478, 2: 259, 6: 235, 8: 16, 0: 12})
Counter({4: 484, 6: 246, 2: 235, 8: 24, 0: 11})
Counter({4: 507, 2: 229, 6: 228, 0: 22, 8: 14})
Counter({4: 513, 2: 234, 6: 226, 8: 16, 0: 11})

Here we can see, in every iteration (each with 1000 trials) that a guess of 4 came up the most, followed by 2 and 6.


4. Change the number of trials from 8 to 10. (You will need to specify a new "correct" array.) Rerun your experiments from #3. How many times do you get >= 8 correct? 

>>> for i in range(10):
...   run_experiments(ten_cups,1000)
...
Counter({4: 404, 6: 396, 2: 96, 8: 93, 10: 6, 0: 5})
Counter({6: 413, 4: 392, 8: 116, 2: 72, 10: 5, 0: 2})
Counter({6: 406, 4: 394, 8: 98, 2: 95, 0: 5, 10: 2})
Counter({6: 409, 4: 374, 8: 105, 2: 102, 10: 8, 0: 2})
Counter({4: 422, 6: 365, 8: 103, 2: 102, 0: 5, 10: 3})
Counter({6: 398, 4: 386, 8: 106, 2: 105, 0: 3, 10: 2})
Counter({6: 415, 4: 411, 8: 93, 2: 74, 10: 4, 0: 3})
Counter({4: 397, 6: 389, 2: 104, 8: 100, 10: 7, 0: 3})
Counter({6: 403, 4: 390, 8: 106, 2: 91, 10: 6, 0: 4})
Counter({6: 426, 4: 370, 2: 103, 8: 90, 10: 6, 0: 5})

Here 4 and 6 are the most common guesses. 
>>> sum = 0
>>> for i in range(10):
...   n=run_experiments(ten_cups,1000)
...   sum = sum + n[8] + n[10]
...
>>> sum
1020

So here we can see we get >= 8 guesses correctly 1020 out of 10000 trials.  

5. If someone tells you they can tell the difference between Gimme's Espresso Blend and Holiday Blend, what experiment would you design to test their ability? How many cups would you ask them to taste, and what would you tell them about the experiment? How many cups would you need them to taste for you to be satisfied that they really can tell the difference even if they make a mistake?

I'm no Coffee connosieur but but I'm pretty sure these blends might be difference colors. So first I would have them in a non-see-through
cup with a lid (almost like a sippy cup) so that they couldn't see what they were drinking. Then I would give them 8 of these non-see-through cups.
I would give one cup of either Expresso or Holiday Blend chosen randomnly. I would take note of their answer. I would then give them one of the opposite blend,
followed by one of the opposite blend again, and then the original blend. If they were correct for all four, I would tell them that they were incorrect and tell them 
they have one more chance.
I would then repeat this in the same order, so they have gone through all 8 cups. 
By now they know both tastes. This experiment will allow us to see if they were just randomnly guessing, or if they actually did know their Coffee.
If they did know, then they would get all 8 cups correct, since they would trust their gut and still guess the correct blend. Otherwise they would likely change their answers. 
Overall this test might not actually work, but it would probably satisfy me, since some people are caffeine addicted and probably do know their coffee. 


"""

from collections import Counter
import random

six_cups = [1,0,0,0,1,1]
eight_cups = [1, 1, 0, 0, 1, 0, 1, 0]
ten_cups = [1,0,1,1,1,0,0,0,1,0]

## Simulate a random guess with an equal number of positives/negatives
def guess_equal(correct):
    n = len(correct)
    
    ## Make a copy of the correct list, then shuffle it
    guess = list(correct) 
    random.shuffle(guess)
    
    #print(correct)
    #print(guess)
    
    correct_guesses = 0
    for i, j in zip(correct, guess):
        if i == j:
            correct_guesses += 1
            
    return correct_guesses

def guess_randomly(correct):
    n = len(correct)
    
    ## Simulate purely random guessing
    guess = []
    for i in range(n):
        guess.append(random.randint(0,1))
    
    #print(correct)
    #print(guess)
    
    correct_guesses = 0
    for i, j in zip(correct, guess):
        if i == j:
            correct_guesses += 1
            
    return correct_guesses


def run_experiments(correct, num_trials, guessr=None):
    results = []

    # Write your "do num_trials experiments" for-loop here:
    # if not none (true) run guess_randomly rather than guess_equal
    if guessr is not None:
        for i in range(num_trials):
            n = guess_randomly(correct)
            results.append(n)
        results = Counter(results)
    else:
        for i in range(num_trials):
            n = guess_equal(correct)
            results.append(n)
        results = Counter(results)

    return results
