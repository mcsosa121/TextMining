"""

We want to find interpretable, low-dimensional models of documents. What does
a Singular Value Decomposition do, and why might it be useful?

A singular value decomposition is a unique breakup of a matrix into three diffrent matrices,
a U matrix, a diagnol S matrix containing the singular values of the matrix, and a V' (transpose)
matrix such that if the matrix is A, the svd computes A=USV' . The singular values of A are obtained
by taking the square roots of the eigenvalues of A'A. 
The singular value decomposition is useful because it solves the linear least squares minimization 
problem. Additionally the SVD is useful in characterising certain properties of the matrix like its rank.

You will need the novels-gutenberg directory in the current directory.

1. Working with vectors and matrices. At the >>> python prompt, create this matrix:

  x = np.array([[1,2,3], [1,2,3], [1,2,3]])

Now describe the value of the following expressions. Write your answers between lines.

Note: Python is zero indexed so the topmost left element in the array is at index (0,0)

a. x[1,1]
----------------------------
Grabs the element at index (1,1) in the array which is 2

b. x[:,1]
----------------------------
Grabs the column vector at column index 1 which is [[2],[2],[2]]

c. x[1,:]
----------------------------
Grabs the row vector at row index 1, which is [1,2,3]

d. np.diag(x[1,:])
----------------------------
Creates a diagnol matrix (A matrix that has zero's everywhere except the diagnol) from the row vector at
row index 1. The values of the inputted vector are the values of the diagnol of the created matrix.
For example x[1,:] = [[2],[2],[2]] => np.diag(x[1,:]) =  np.diag([[2],[2],[2]]) = np.array([2,0,0],[0,2,0],[0,0,2])

e. x.T
----------------------------
Takes the transpose of x. The transpose is the index of each element swtiched (i.e (a,b)->(b,a))
ex. x.T = (np.array([[1,2,3], [1,2,3], [1,2,3]])).T = np.array([[1,1,1],[2,2,2],[3,3,3]])

f. x[1,:].dot( x[2,:] )
----------------------------
Takes the dot product of a row vector and a column vector. The dot product is \sum_{i=0}^{k} (x_i*y_i). Here our result 
returns 14 in this case.

g. x[:,0].dot( x[:,1] )
----------------------------
Takes the dot product of a column vector and a row vector. The result for this matrix is 6. 

h. x.dot( x[:,1] )
----------------------------
This dots the entire matrix x, with one of its column vectors. Essentially this acts as matrix multiplication. 
The resulting matrix is 3x3*3*1=3x1 and is [[3],[3],[3]]

i. x.dot( x )
----------------------------
This is dotting x with itself performing x*x. The result of this is
array([[ 6, 12, 18],
       [ 6, 12, 18],
       [ 6, 12, 18]])

j. x.dot( np.diag(x[1,:]) )
----------------------------
np.diag is creating a diagnol matrix, with the elements across the diagnol coming from a row vector x[1,:]
The operation is then dotting x with the new diagnol matrix performing x*diag(x[1,:]) which evaluates to 
array([[1, 4, 9],
       [1, 4, 9],
       [1, 4, 9]])

k. x ** 2
----------------------------
This performs elementwise power operations on each element of the matrix. Aka, each element of the matrix is squared
resulting in 
array([[1, 4, 9],
       [1, 4, 9],
       [1, 4, 9]])

l. x - np.array([1,2,3])
----------------------------
This operation does elementwise subtraction. However, since the element in question in a array, then 
for each row vector in x, np.array([1,2,3]) is subtracted from that vector resulting in
array([[0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]])

m. x - np.array([1,2,3])[:,None]
----------------------------
The command
np.array([1,2,3])[:,None] yeilds
array([[1],
       [2],
       [3]])
This is because it is taking each row and placing it in its own indavidual row.
array([[ 0,  1,  2],
       [-1,  0,  1],
       [-2, -1,  0]])
The resulting matrix multiplication is show above.abs

n. np.sum(x)
----------------------------
This takes the sum of all of the elements within the matrix. The result is 18.

o. np.sum(x, axis=0)
----------------------------
The additional argument specifies if you want only a certain axis to be summed. 
axis=0 means that the matrix is summed along columns resulting in array([3, 6, 9])

p. np.sum(x, axis=1)
----------------------------
On the other hand axis=1 means that the matrix is summed across rows resulting in array([6, 6, 6])

q. np.sum(x ** 2, axis=1)
----------------------------
Squares each element before adding it to the rowsum. The result is array([14, 14, 14])

r. np.sqrt(np.sum(x ** 2, axis=1))
----------------------------
Same as above but also eakes the elementwise squareroot resulting in array([3.74, 3.74, 3.74])

2. The code below runs an SVD. Describe some properties of these matrices.

Properties of the SVD are that the V is orthogonal (thus the matrix A maps the orthonormal columns of V into 
the orthonormal columns of U). Like described above, these matrices are useful in solving least squares 
problems such as in image reconstruction. Essentially the SVD is reducing the dimension of the original matrix. 
So an image once of rank k could be approximated by l where l<<k (in some cases, not all).

3. Describe the properties of the `weights` vector.

The weights vector is the singular values of A. The singular values are the square roots of the eigenvalues of AA'. 
These singular values are important in determining things like preferences among uses when your data is in high dimensional space.

4. What is the first column of the file_vectors and word_vectors (transpose) matrix representing?

First vector whose weight is 0.7785 .... has more value over other vectors with weights. We can think 
of the first column as a "best fit" for approximating the novel.

4. What is the second column of the file_vectors and word_vectors (transpose) matrix representing?

The second column, represents the same thing, except it is the second best approximation to the novel.
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

##
weighted_word_vectors = word_vectors.dot( np.diag(np.sqrt(weights)) )
weighted_file_vectors = file_vectors.dot( np.diag(np.sqrt(weights)) )
