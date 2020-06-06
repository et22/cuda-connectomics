# Ethan Trepka
# preprocessing microns L2 L3 dataset
# file_num =0 for simple file with ~500 vertices, 1 for file with million vertices


def process_microns_undirected(file_num):
    file_names = ["microns_messy_data/soma_subgraph_synapses_spines_v185.csv", "microns_messy_data/pni_synapses_v185.csv"]
    file = file_names[file_num]
    current_dictionary = {}
    in_file = open(file, "r")
    count = 0
    numEdges = 0
    for line in in_file:
        if not count == 0:
            split_line = line.split(',')
            if split_line[1] not in current_dictionary:
                current_dictionary[split_line[1]] = set([split_line[2]])
                numEdges += 1
            else:
                list_len = len(current_dictionary[split_line[1]])
                current_dictionary[split_line[1]].add(split_line[2])
                if not list_len == len(current_dictionary[split_line[1]]):
                    numEdges += 1
                if split_line[2] not in current_dictionary:
                    current_dictionary[split_line[2]] = set()
            # making the graph undirected
            if split_line[2] not in current_dictionary:
                current_dictionary[split_line[2]] = set([split_line[1]])
                numEdges += 1
            else:
                list_len = len(current_dictionary[split_line[2]])
                current_dictionary[split_line[2]].add(split_line[1])
                if not list_len == len(current_dictionary[split_line[2]]):
                    numEdges += 1
                if split_line[1] not in current_dictionary:
                    current_dictionary[split_line[1]] = set()
        count += 1
    in_file.close()
    return current_dictionary, numEdges


def write_adjacency_matrix(dictionary):
    matrix = [[0 for x in range(len(dictionary))] for y in range(len(dictionary))]
    key_to_row = {}
    count = 0
    for key in dictionary:
        key_to_row[key] = count
        count += 1
    for key in dictionary:
        for value in dictionary[key]:
            matrix[key_to_row[key]][key_to_row[value]] = 1
    out_file = open("microns_cleaned_data/microns_out.txt", "w")
    for row in range(len(dictionary)):
        row_string = ""
        for col in range(len(dictionary)):
            row_string += str(matrix[row][col])
            row_string += " "
        out_file.write(row_string)
        out_file.write("\n")
    out_file.close();


def write_csr_matrix(dictionary, num_edges):
    key_to_row = {}
    count = 0
    out_file_data = open("microns_cleaned_data/microns_out_data.txt", "w")
    data_string = str(len(dictionary)) + " " + str(num_edges)
    out_file_data.write(data_string)
    out_file_data.close()
    out_file = open("microns_cleaned_data/microns_out.txt", "w")
    for key in dictionary:
        key_to_row[key] = count
        count += 1
    for key in dictionary:
        for value in dictionary[key]:
            entry_string=str(key_to_row[key]) + " " + str(key_to_row[value]) + " 1"
            out_file.write(entry_string)
            out_file.write("\n")
    out_file.close()
    print(len(dictionary))

