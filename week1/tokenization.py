### On Monday we talked about how encodings allow us to map
###  bytes to characters. In this lesson we'll look at finding
###  meaningful groups of characters.

### In the git repo you will find six files containing
###  sample text. Each is selected to highlight issues in
###  tokenization. I've included a template for the first
###  one, which we will modify together in class.

import re, sys

## SAMPLE 1

## Here's an example of a simple pattern defining a word token.
print("Sample 1")
word_pattern = re.compile("([\w]+[,][\w]+(?= )|[\w]+[-\w]*|[\w]+(?=,)|[\w]+)")
with open("sample1.txt", encoding="utf-8") as file:

    ## This block reads a file line by line.
    for line in file:
        line = line.rstrip()

        ## This converts a string (line) into a list (tokens)
        tokens = word_pattern.findall(line)

        print(tokens)
print("---------------------------------------------------------------------")
### [Discuss sample 1 here.]
### This is the default example. The sentences are pretty simple to tokenize.
### I changed the regex to accomodate for words with dashes and numbers with
### commas but not words or numbers ending with commas

## SAMPLE 2
print("Sample 2")
word_pattern = re.compile("([\w]+[,][\w]+(?= )|[\w]+[-\w]*|[\w]+(?=,)|🐱|[\w]+)")
with open("sample2.txt", encoding="utf-8") as file:

    ## This block reads a file line by line.
    for line in file:
        line = line.rstrip()

        ## This converts a string (line) into a list (tokens)
        tokens = word_pattern.findall(line)

        print(tokens)
print("---------------------------------------------------------------------")


### Sample two was similar. I didn't want to include things like ! but
### I did want to include emojis. Since emoji's range from a wide variety
### if we know what emoji's we are using beforehand we can just include
### them in the regex.


## SAMPLE 3

### [Copy and modify code as necessary here.]
print("Sample 3")
word_pattern = re.compile("([\w]+[,][\w]+(?= )|[\w]+[-\w]*|[\w]+(?=,)|🐱|[\w]+)")
with open("sample3.txt", encoding="utf-8") as file:

    ## This block reads a file line by line.
    for line in file:
        line = line.rstrip()

        ## This converts a string (line) into a list (tokens)
        tokens = word_pattern.findall(line)

        print(tokens)
print("---------------------------------------------------------------------")
### Nothing too special here. The one case I was interested in was ")-Cas9"
### but after looking it up Cas9 is its own abbreviation so this regex works.

## SAMPLE 4
print("Sample 4")
word_pattern = re.compile("([\w]+[,][\w]+(?= )|[\w]+[-\w]*|[\w]+(?=,)|🐱|[\w]+)")
with open("sample4.txt", encoding="utf-8") as file:

    ## This block reads a file line by line.
    for line in file:
        line = line.rstrip()

        ## This converts a string (line) into a list (tokens)
        tokens = word_pattern.findall(line)

        print(tokens)
print("---------------------------------------------------------------------")
### Tokenizes Spanish and extracts the words well.



## SAMPLE 5
print("Sample 5")
word_pattern = re.compile("([\w]+[,][\w]+(?= )|[\w]+(?=,)|[\w]+)")
with open("sample5.txt", encoding="utf-8") as file:

    ## This block reads a file line by line.
    for line in file:
        line = line.rstrip()

        ## This converts a string (line) into a list (tokens)
        tokens = word_pattern.findall(line)

        print(tokens)
print("---------------------------------------------------------------------")

### [Discuss sample 5 here.]


## SAMPLE 6
print("Sample 6")
word_pattern = re.compile("([\w]+[,][\w]+(?= )|[\w]+(?=,)|[\w]+)")
with open("sample6.txt", encoding="utf-8") as file:

    ## This block reads a file line by line.
    for line in file:
        line = line.rstrip()

        ## This converts a string (line) into a list (tokens)
        tokens = word_pattern.findall(line)
        for t in tokens:
            for i in t:
                print(i.encode("utf-8"))

        print(tokens)
print("---------------------------------------------------------------------")

### [Discuss sample 6 here.]
