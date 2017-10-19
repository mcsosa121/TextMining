"""

We want to find interpretable, low-dimensional models of documents. What does 
a Singular Value Decomposition do, and why might it be useful?


You will need the novels-gutenberg directory in the current directory.

1. Working with vectors and matrices part II. At the >>> python prompt, create this matrix:

  x = np.array([[ 0.37962213,  1.21124263,  0.56771852],[ 1.36941388,  2.62035395,  2.25421422],[ 0.72534686,  1.55442357,  1.90488544]])

Now create the SVD:

  (U, s, V) = np.linalg.svd(x)

Now describe the value of the following expressions. What is the code doing?
Why do you get the results you see? Write your answers between lines.

a. U
array([[-0.28370447,  0.7019116 ,  0.65332371],
       [-0.79210722,  0.21244662, -0.57221726],
       [-0.54044235, -0.67984302,  0.49571719]])
The code is computing the left singular vectors of the matrix, where each column can be seen as a 
decreasing series of approximations.

b. s
array([ 4.68944102,  0.54874723,  0.18848219])
This is an array of singular values of x which are the square roots of the eigenvalues of AA'

b*. V
V are the right singular vectors that are constructed from x. 


c. U[:,0]
Prints a column vector of the best approximation. 
array([-0.28370447, -0.79210722, -0.54044235])

d. U[:,0].dot(U[:,0])
Dots the first column vector with itself. The result is 1 implying 
that the matrix is orthogonal which it is because the columns are
orthonormal.

e. U[:,0].dot(U[:,1])
Dots the first column vector with the second column (aka the second
best approximation). This returns a very small value
-4.4408920985006262e-16 which is close to zero. This means that 
these column vectors are almost perpindicular to eachother.

f. V[0,:].dot(V[0,:])
Is the first row vector (we do this because we want the transpose of V)
dotted with itself. This results in 1 implying the matrix is orthogonal 
because the columns of V' are orthonormal.

g. V[0,:].dot(V[1,:])
Likewise here we receive a number very close to zero, showing that 
these column vectors of V' (or row vectors of V) are almost perpindicular. 
-1.1102230246251565e-16 is the result.

# In addition to your description, compare the following expressions to the original matrix x
h. np.outer( U[:,0], V[0,:] )
Results in 
array([[ 0.09585582,  0.19718374,  0.18005126],
       [ 0.26763093,  0.55054003,  0.50270589],
       [ 0.18260039,  0.37562484,  0.34298836]])

i. np.outer( U[:,0], V[0,:] ) * s[0]
Results in 
array([[ 0.44951023,  0.92468153,  0.84433979],
       [ 1.25503944,  2.58172498,  2.35740961],
       [ 0.85629376,  1.76147053,  1.60842366]])

j. np.outer( U[:,0], V[0,:] ) * s[0] + np.outer( U[:,1], V[1,:] ) * s[1]
Results in 
array([[ 0.49461965,  1.17042541,  0.55119717],
       [ 1.26869265,  2.65610395,  2.26868453],
       [ 0.81260261,  1.52345301,  1.89234967]])

k. np.outer( U[:,0], V[0,:] ) * s[0] + np.outer( U[:,1], V[1,:] ) * s[1] + np.outer( U[:,2], V[2,:] ) * s[2]
Results in 
array([[ 0.37962213,  1.21124263,  0.56771852],
       [ 1.36941388,  2.62035395,  2.25421422],
       [ 0.72534686,  1.55442357,  1.90488544]])
Which is the original x. This is essentially showing that by taking combinations of the columns and rows of the left and right 
singular vectors and using the weights we can approximate the original matrix as well as reconstruct it.   


2. Imagine that half your documents are in English and half are in French. What might be different about the `weights` array?
Some weights may approximate the english novels better, while others may approximate the french novels better.
Words in english may be negatively connotated but similar sounding 

3. Use the `sort_vector()` function to examine the word and document vectors. Find some examples that are surprising, interesting, or confusing.
(-0.58784424104047894, 'the')
(-0.36017932948639075, 'and')
(-0.30653754561790902, 'of')
(-0.29116758103800722, 'to')
(-0.23100808469243636, 'a')
(-0.19142062600828455, 'I')
(-0.16835086229106269, 'in')
(-0.13434607749615149, 'that')
(-0.13041940269807975, 'was')
(-0.12187193816277338, 'he')
(-1.629524590526345e-05, 'Plumet')
(-1.6016220295213072e-05, 'Champs-Élysées')
(-1.5969340987158217e-05, 'Théodule')
(-1.5643436069052909e-05, 'Petit-Picpus')
(-1.5105357932930828e-05, 'grape-shot')
(-1.499162623284239e-05, 'Toussaint')
(-1.4665721314737079e-05, 'Babet')
(-1.4665721314737079e-05, 'Chanvrerie')
(-1.4339816396631826e-05, 'Boulatruelle')
(-1.4339816396631826e-05, 'Champmathieu')

4. Do you trust 2D visualizations of documents? Why or why not? What would you want to know before using one?
I trust 2D visualizations to tell me some of the information about the document, but must also remember that 
it is an approximation to the true form of the data which lies in n dimensional space. 
However, before wanting to use one, I would want to make sure that I know linear algebra solidly as well 
as the algorithms behind the method. That way I would be able to properly interpret the results of the data

"""

import re, sys
import numpy as np
from collections import Counter

word_pattern = re.compile("\w[\w\-\']*\w|\w")

novel_metadata = {} # <- dict of dicts
novel_counts = {} # <- dict of Counters
word_counts = Counter()

## Open the metadata file and read the header line
metadata_reader = open("novels-gutenberg/metadata.tsv")
metadata_fields = metadata_reader.readline().rstrip().split("\t")

## Read the rest of the lines of the file
for line in metadata_reader:
    
    ## Get the next line of the metadata file, and split it into columns
    fields = line.rstrip().split("\t")
    
    ## Convert the list of field values into a map from field names to values
    metadata = dict(zip(metadata_fields, fields))
    
    if not "Author" in metadata:
        metadata["Author"] = "Unknown"
    
    ## Save it for later with the filename as key
    filename = "novels-gutenberg/text/pg{}.txt".format(metadata["ID"])
    novel_metadata[filename] = metadata

    ## Now count the words in the novel
    counter = Counter()
    with open(filename, encoding="utf-8") as file:
        
        ## This block reads a file line by line.
        for line in file:
            line = line.rstrip()
            
            tokens = word_pattern.findall(line)
            
            counter.update(tokens)
    
    ## And save those counts for later
    novel_counts[filename] = counter
    
    ## Record the total number of times each word occurs
    word_counts.update(counter)

## All the distinct word types in descending order by frequency
vocabulary = [x[0] for x in word_counts.most_common()]

vocabulary = vocabulary[0:10000]
reverse_vocabulary = {}
for word_id, word in enumerate(vocabulary):
    reverse_vocabulary[word] = word_id

## All the filenames
filenames = list(novel_counts.keys())
titles = [novel_metadata[id]["Title"] for id in filenames]

## Allocate a matrix with two rows for every file and one column for every word type
file_word_counts = np.zeros([ len(filenames), len(vocabulary) ])

## Convert a map of file-level counters to a single matrix
## We'll use two index variables, file_id and word_id. These will be 
##  *numbers*, not strings, that point to a string in either of
##  the two arrays.
for file_id in range(len(filenames)):
    counter = novel_counts[ filenames[file_id] ]
    
    for word_id in range(len(vocabulary)):
        file_word_counts[file_id,word_id] = counter[ vocabulary[word_id] ]
    
    ## Normalize for length
    file_word_counts[file_id,:] /= np.sum(file_word_counts[file_id,:])

## Run the singular value decomposition
(file_vectors, weights, word_vectors) = np.linalg.svd(file_word_counts, full_matrices=False)

## transpose word vectors
word_vectors = word_vectors.T

## Absorb the factor weights into the matrices
weighted_word_vectors = word_vectors.dot( np.diag(np.sqrt(weights)) )
weighted_file_vectors = file_vectors.dot( np.diag(np.sqrt(weights)) )

## Write data to files, which we can load with R

with open("file_vectors.tsv", "w") as out:
    for i in range(len(file_vectors[0,:])):
        out.write("V{}\t".format(i))
    out.write("Title\n")
    
    for file_id in range(len(filenames)):
        for i in range(len(file_vectors[file_id,:])):
            out.write("{:.6f}\t".format(file_vectors[file_id,i]))
        out.write("{}\n".format(titles[file_id]))

with open("word_vectors.tsv", "w") as out:
    for i in range(len(word_vectors[0,:])):
        out.write("V{}\t".format(i))
    out.write("Word\n")
    
    for word_id in range(len(vocabulary)):
        for i in range(len(word_vectors[word_id,:])):
            out.write("{:.6f}\t".format(word_vectors[word_id,i]))
        out.write("{}\n".format(vocabulary[word_id]))


## Pass in a vector and a list of strings that define a meaning for each
##  element in the vector. Sort the elements by value and print the top
##  and bottom strings.
def sort_vector(v, names):
    sorted_list = sorted(list(zip(v, names)))
    for pair in sorted_list[:10]:
        print(pair)
    for pair in sorted_list[-10:]:
        print(pair)


