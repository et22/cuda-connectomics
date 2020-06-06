# Ethan Trepka
# test graphs to demonstrate bfs correctness
import random


def generate_test_0():
    dictionary = {"1": set(["2"]), "2": set(["1", "3"]), "3": set(["2"])}
    number_of_edges = 4

    return dictionary, number_of_edges


# simple, likely highly connected graph
def generate_test_1():
    dictionary = {"1": set(), "2": set(), "3": set(), "4": set(), "5": set()}
    number_of_edges = 0
    for key in dictionary:
        number = random.randint(0, 2)
        for j in range(number):
            i = random.randint(1, 5)
            if str(i) not in dictionary[key]:
                dictionary[key].add(str(i))
                number_of_edges += 1
            if key not in dictionary[str(i)]:
                dictionary[str(i)].add(key)
                number_of_edges += 1

    return dictionary, number_of_edges


# less dense graph
def generate_test_2():
    dictionary = {"1": set(), "2": set(), "3": set(), "4": set(), "5": set(), "6": set(), "7": set(), "8": set(),
                  "9": set(), "10": set()}
    number_of_edges = 0

    for key in dictionary:
        number = random.randint(0, 1)
        for j in range(number):
            i = random.randint(1, 10)
            if str(i) not in dictionary[key]:
                dictionary[key].add(str(i))
                number_of_edges += 1
            if key not in dictionary[str(i)]:
                dictionary[str(i)].add(key)
                number_of_edges += 1

    return dictionary, number_of_edges


# graph guaranteed to have some empty edge sets, some vertices that have only one edge,
# and others with more than one edge
def generate_test_3():
    dictionary = {"1": set(), "2": set(), "3": set(), "4": set(), "5": set(), "6": set(), "7": set(), "8": set(),
                  "9": set(), "10": set(), "11": set(), "12": set()}
    dictionary["1"].add("2")
    dictionary["2"].add("1")
    dictionary["11"].add("12")
    dictionary["12"].add("11")
    dictionary["11"].add("1")
    dictionary["1"].add("11")
    dictionary["5"].add("6")
    dictionary["6"].add("5")
    number_of_edges = 8

    return dictionary, number_of_edges
