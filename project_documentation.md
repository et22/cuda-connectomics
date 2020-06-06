**GPU Programming and Higher Performance Computing --- Final Project
Proposal **

**Ethan Trepka**

**Overview**

<https://www.tandfonline.com/doi/pdf/10.1080/00018730601170527?needAccess=true>

Connectomes are architectural maps of all neurons and synapses in the
brain. Connectomes can be represented as directed graphs with edge
weights representing either the number of synapses or sum of synaptic
size. This directed graph can be represented as a sparse adjacency
matrix where each row or column is a neuron and the value of each
element is the number of synapses between the row and column neuron.

For my final project, I want to implement various graph algorithms in
CUDA on different connectome graphs that could produce interesting and
biologically meaningful results. For example, examining strongly
connected components across the brain or within brain regions could
elucidate interesting feedback pathways. Similarly, using DFS to
determine unconnected nodes in the graph could be used to identify
errors in graph construction since the graphs were constructed in an
automated fashion. Finally, after segmenting the data by brain region,
within region parameters can be computed such as minimum vertex
coloring, minimum edge coloring and more.

I may not be able to do all of these things over the next two weeks, but
I hope to at least implement a few of the above graph algorithms
efficiently on CUDA and test them on at least one of the datasets
described below.

**Datasets**

**1.L2/L3 Mouse Connectome data from MiCRONs project** (\~2000 manually
verified synapses, 2.3 million automatically predicted synapses)

This data consists of results from parsing EM images of layers 2 and 3
of the primary visual cortex, V1. The L2/L3 connectome dataset contains
the following data for all pyramidal-to-pyramidal cell synapses in the
image volume:

-synaptic spine volume

-presynaptic cell id

-post synaptic cell id

-presynaptic terminal location

-post synaptic dendritic spine location

<https://microns-explorer.org/phase1>

**2. Whole brain drosophila connectome data** (\~244 million
automatically predicted synapses, 96% accuracy)

The drosophilia connectome dataset contains the following data for all
cell synapses in the image volume:

-presynaptic cell id

-post synaptic cell id

-presynaptic terminal location

-post synaptic dendritic spine location

<https://neuprint.janelia.org/>

<https://github.com/flyconnectome/fafbseg-py/blob/query-synapses/notebooks/Synaptic_Partner_Predictions_in_FAFB.ipynb>

**Challenges**

-Parsing datasets; For ease of data usage, I may use PyCUDA which just
requires writing kernel functions in CUDA

-The main problem with the datasets that I currently have is that there
are no architectural labels i.e. we know the absolute position of each
neuron, but we don’t know what brain region the neuron is in. This will
hinder the interpretability of results, but not the construction of the
algorithms.

**A few ideas for algorithms and Papers**

Graph algorithm implementations in CUDA -

<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.4206&rep=rep1&type=pdf>

Fly connectome preprint –

<https://www.biorxiv.org/content/10.1101/2020.01.21.911859v1.full>
