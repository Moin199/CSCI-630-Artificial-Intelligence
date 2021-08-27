import sys


def modify_word(word, successor: list, words: list):
    """

    :param word: single word
    :param successor: succcessor of the word to fill
    :param words: words list
    :return: successor list
    """
    for index in range(0, len(word)):
        for i in range(ord('a'), ord('z')):
            new_str = (word[0:index] + chr(i) + word[index + 1:len(word)]).lower()
            if new_str != word:
                if new_str in words:
                    # if new_str == final:
                    #     return successor
                    successor.append(new_str)
    return successor


def get_successor(initial, words):
    """

    :param initial: initial word
    :param words: word list
    :return: successor of the word
    """
    successor = modify_word(initial, [], words)
    return successor


def create_dict(words, initial):
    """

    :param words: list of words
    :param initial: initial word
    :return: return a dictionary of words with its successor as a value of the word key
    """
    word_dict = {}
    for word in words:
        # Inserting only the same length words in the dictionary
        if len(word) == len(initial):
            word_dict[word] = get_successor(word, words)
    return word_dict


def findShortestPath(start, end, global_dict):
    """
    Find the shortest path, if one exists, between a start and end vertex
    :param start: initial word
    :param end : final word
    :param global_dict: dictionary containing links between words

    :return: A list of words from start to end, if a path exists,
        otherwise None
    """
    # queue to store the list of words visited in breadth first manner
    queue = []
    # append the initial word
    queue.append(start)

    # The predecessor dictionary maps the current word to its
    # parent.  This collection serves as both a visited
    # construct, as well as a way to find the path
    parent = dict()
    parent[start] = None  # add the start word with no predecessor

    # Loop until either the queue is empty, or the final is encountered
    while len(queue) > 0:
        current = queue.pop(0)
        if current == end:
            break
        try:
            # map a particular list for a key in dictionary
            # if word is univisted, then make the current list a parent of next word
            for neighbor in global_dict[current]:
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
        except Exception as e:
            print(e)
    # If the final is in parent a path was found
    if end in parent:
        path = []
        current = end
        # start looping from end
        # until the initial word is not found,prepend the path list with the current word
        # make the successor of the current word, the current word
        while current != start:
            path.insert(0, current)
            current = parent[current]
        path.insert(0, start)
        return path
    else:
        return None


def main():
    """
    Main function to collect input
    and call other functions
    Prints the output
    """
    dict_path = sys.argv[1]
    initial = sys.argv[2]
    final = sys.argv[3]
    with open(dict_path) as f:
        words = f.read().lower().split("\n")
    if len(initial) != len(final):
        print("No solution")
    else:
        global_dict = create_dict(words, initial)
        path = findShortestPath(initial, final, global_dict)
        if path is not None:
            for index in path:
                print(index)
        else:
            print("No solution")


if __name__ == '__main__':
    # call main function to execute
    main()
