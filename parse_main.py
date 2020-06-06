# Ethan Trepka
# Calls parse functions in microns_data_preprocessing to create matrices to read in C
from microns_data_preprocessing import *


dictionary, edges = process_microns_undirected(1)
write_csr_matrix(dictionary, edges)
