"""

Today we'll finish up our new corpus and start thinking about clustering.
Specifically, what are the issues in agglomerative clustering?

Class notes:
n-1 elements in the new list. need to do n(n-1)/2

have two similar elements, merge using jacard similarity, how can you use this in future comparisons.

first line in new metadata is headers. So you must be careful to know what format that you're using

QUESTIONS

1. This code is currently assuming the structure of the OTA corpus. Convert it to
use the structure of our new corpus.

The new corpus structure is "ID	Title	Author	Year	NetID	Original Language	Genre	Author Age	Author Gender	Source". So we have to change
the metadata_fields variable to reflect this change. Need to also ignore the first line which is defining the metadata fields. Also need to create
the filenames using the "pg{}.txt".format(field["ID"]). Also had to check if the file was actually there since some people had thier info in the metadata
but didn't actually have their text in the collection. This was sad because I was really looking forward to reading "The Romance of Lust: A Classic Victorian erotic novel".
I also had to change other various parts of the code.

2. Agglomerative clustering merges the closest pair of clusters. What should happen then? List possible options and their advantages / disadvantages.

Once the algorithm is finished, AC has put all of the data into a single cluster, with each level of the resulting tree being diffrent parts of the data.
Once you have this single clustering, you can go down the tree and choose groups appropriately. For example, if clusters represent genres, the second level
represents a certain number of overall genres. As you go down the tree further, the genres are broken up into further subgenres. Now the user has a few diffrent
options with regards to intercluster similarities, (1) The similarity of the closest pair, (2) the similarity of the furthest pair, or (3) average similarity
between groups. 1 is not good because you can merge groups too early. Similarly (2) might not merge groups because they're too far apart. (3) has downsides
because taking the average can possibly change results.

3. What makes agglomerative clustering fast or slow?

Because there are n-1 elements that you have to compare to, then the total number of elements that you have to compare to
can be shown by n(n-1)/2 which means that the code is O(n^2) in complexity at least. However compared to having to a  O(n!)
runtime in which all possible pairs are considered, this is great :) .

4. What do you notice about the distances between merged clusters in this implementation as the algorithm proceeds?

The distances between merged clusters goes up as the algorithm runs because the AC is merging bigger groups.

5. What do you think of agglomerative clustering? Is it reasonable? Is the output useful? Why or why not? What would you prefer?

I think it is a useful tool for data analysis. One of the advantages of using agglomerative clustering is that AC only requires a measure of similarity between groups
of data, as opposed to K-means clustering which requires distance measure between data, initial assignment of data to clusters, and a number of clusters known.
The output is useful for summarizing results. However, you must also be careful to take into account the diffrent methods you use and realize that the methods that you take
can result in diffrent outputs. I would prefer to use this, but always note that we made the decisions that we did for certain reasons and make that clear.

"""

import re, sys
import numpy as np
from collections import Counter

desired_clusters = 40

word_pattern = re.compile("\w[\w\-\']*\w|\w")

novel_metadata = {} # <- dict of dicts
novel_counts = {} # <- dict of Counters
word_counts = Counter()

metadata_fields = ["ID", "Title", "Author", "Year", "NetID", "Original Language", "Genre", "Author Age", "Author Gender", "Source"]

## Read a collection, one .txt file per document, with metadata from a .tsv.
for line in open("metadata.tsv"):
    ## Get the next line of the metadata file, and split it into columns
    fields = line.rstrip().split("\t")

    # need to ignore the first line
    if fields[0]!="ID":
        ## Convert the list of field values into a map from field names to values
        metadata = dict(zip(metadata_fields, fields))

        if not "Author" in metadata:
            metadata["Author"] = "Unknown"

        ## Save it for later with the filename as key
        filename = "text/pg{}.txt".format(metadata["ID"])
        novel_metadata[filename] = metadata

        ## Now count the words in the novel
        counter = Counter()
        try:
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
        except FileNotFoundError:
            # need to remove the offending text.
            del novel_metadata[filename]

## All the distinct word types in one list
vocabulary = list(word_counts.keys())

## All the filenames
filenames = list(novel_counts.keys())

## Allocate a matrix with two rows for every file and one column for every word type
file_word_counts = np.zeros([ 2 * len(filenames), len(vocabulary) ])

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

pairwise_distances = []

for file_a in range(len(filenames)):

    ## The original files will be normalized, but the merged
    ##  nodes will not, so make sure everything adds up to 1.0
    counts_a = file_word_counts[file_a,:] / np.sum(file_word_counts[file_a,:])
    for file_b in range(file_a + 1, len(filenames)):

        counts_b = file_word_counts[file_b,:] / np.sum(file_word_counts[file_b,:])

        diff_a_b = np.sum( np.abs( counts_a - counts_b ) )
        pairwise_distances.append( (file_a, file_b, diff_a_b) )

## Sort in ascending order by similarity
pairwise_distances.sort(key = lambda x: x[2])

num_nodes = len(filenames)
parents = list(range(2 * len(filenames)))

constituent_files = []
for filename in filenames:
    constituent_files.append([filename])

## Now merge until we have only X nodes left
while num_nodes < 2 * len(filenames) - desired_clusters:
    closest_pair = pairwise_distances.pop(0)

    file_a = closest_pair[0]
    file_b = closest_pair[1]

    new_node = num_nodes
    constituent_files.append(constituent_files[file_a] + constituent_files[file_b])

    file_word_counts[new_node,:] = file_word_counts[file_a,:] + file_word_counts[file_b,:]

    print("{:3d}: merging {} and {} ({:.4f})".format(new_node, file_a, file_b, closest_pair[2]))
    parents[file_a] = new_node
    parents[file_b] = new_node

    ## Remove pairs involving merged nodes
#    pairwise_distances = [x for x in pairwise_distances if x[1] != file_a and x[2] != file_a and x[1] != file_b and x[2] != file_b]

    pairwise_distances = [x for x in pairwise_distances if x[0] != file_a and x[1] != file_a and x[0] != file_b and x[1] != file_b]
    #print(pairwise_distances)
    pairwise_distances.sort(key = lambda x: x[2])

    ## Now add the new distances for the merged node
    counts_new = file_word_counts[new_node,:] / np.sum(file_word_counts[new_node,:])
    for file_b in range(num_nodes):
        ## only compare unmerged nodes
        if parents[file_b] == file_b:
            counts_b = file_word_counts[file_b,:] / np.sum(file_word_counts[file_b,:])

            diff_new_b = np.sum( np.abs( counts_new - counts_b ) )
            pairwise_distances.append( (new_node, file_b, diff_new_b) )

    num_nodes += 1

for node in range(num_nodes):
    print(node)

    ## Look for unmerged nodes
    if parents[node] == node:
        for filename in constituent_files[node]:
            print("  {} - {} / {}".format(filename, novel_metadata[filename]["Title"], novel_metadata[filename]["Author"]))
