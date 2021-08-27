import math
import pickle
import random
import sys

DUTCH_COMMON_WORDS = ['ik', 'je', 'het', 'de', 'dat', 'een', 'niet', \
                      'en', 'wat', 'van', 'ze', 'op', 'te', 'hij', 'zijn', 'er', \
                      'maar', 'die', 'heb', 'voor', 'met', 'als', 'ben', 'mijn', 'u', \
                      'dit', 'aan', 'om', 'hier', 'naar', 'dan', 'jij', 'zo', 'weet', \
                      'ja', 'kan', 'geen', 'nog', 'wel', 'wil', 'moet', 'goed', 'hem', \
                      'hebben', 'nee', 'heeft', 'waar', 'nu', 'hoe', 'ga', 'kom', 'uit', \
                      'al', 'jullie', 'zal', 'bij', 'ons', 'gaat', 'hebt', 'meer', \
                      'waarom', 'iets', 'deze', 'laat', 'doe', 'm', 'moeten', 'wie', \
                      'jou', 'alles', 'denk', 'kunnen', 'eens', 'echt', 'weg', \
                      'terug', 'laten', 'mee', 'hou', 'komt','toch', 'zien', 'oké', 'alleen', 'nou', 'dus', 'nooit',
                      'niets', 'zei', \
                      'misschien', 'kijk', 'iemand', 'komen', 'tot', 'veel', \
                      'worden', 'onze', 'mensen', 'zeg', 'leven', 'zeggen', 'weer', \
                      'gewoon', 'nodig','jouw', 'vrouw', 'geld', 'wij', 'twee', 'tijd', 'tegen', 'uw', \
                      'toen', 'zit', 'net', 'weten', 'heel', 'maken', 'wordt', \
                      'dood', 'mag', 'altijd', 'af', 'wacht', 'geef', 'z', 'lk', \
                      'dag', 'omdat', 'zeker', 'zie', 'allemaal', 'gedaan', 'oh', \
                      'dank', 'huis', 'hé', 'zij', 'jaar', 'vader', 'doet', 'zoals',\
                      'hun']

ENGLISH_COMMON_WORDS = ['a', 'about', 'all', 'also', 'and', 'as', 'at', 'be', \
                        'because', 'but', 'by', 'can', 'come', 'could', 'day', 'do', 'even', \
                        'find', 'first', 'for', 'from', 'get', 'give', 'go', 'have', 'he', \
                        'her', 'here', 'him', 'his', 'how', 'I', 'if', 'in', 'into', 'it', \
                        'its', 'just', 'know', 'like', 'look', 'make', 'man', 'many', 'me', \
                        'more', 'my', 'new', 'no', 'not', 'now', 'of', 'on', 'one', 'only', \
                        'or', 'other', 'our', 'out', 'people', 'say', 'see', 'she', 'so', 'some', \
                        'take', 'tell', 'than', 'that', 'their', 'them', 'then', 'there', \
                        'these', 'they', 'thing', 'think', 'this', 'those', 'time', 'to', \
                        'two', 'up', 'use', 'very', 'want', 'way', 'we', 'well', 'what', \
                        'when', 'which', 'who', 'will', 'with', 'would', 'year', 'you', 'your']


def get_input(file_name):
    """
    Get input from file name
    :param file_name:
    :return: X
    """
    X = []
    with open(file_name) as f:
        data = f.readlines()
        for line in data:
            X.append(line.replace("\n", ""))
    return X


def find_duplicates(s):
    """
    Return whether duplicates are present
    :param s:
    :return: Set of words that are duplicates
    """
    return set([i for i in s if s.count(i) > 1])


def isVowel(letter):
    """
    Return whether a letter is Vowel or not
    :param letter:
    :return: true if Vowel
    """
    l = letter.lower()
    return (l == 'a' or l == 'e' or l == 'i' or l == 'o' or l == 'u')


def get_con_count(words):
    """
    Consecutive Consonant Counts
    :param words:
    :return: Average of the words
    """
    count, res = 0, 0
    sum = 0
    for i in range(len(words)):
        for j in words[i]:
            if (isVowel(j) == False):
                count += 1
            else:
                res = max(res, count)
                count = 0
        sum += max(res, count)
    return sum / len(words)


def get_features(X, feature_matrix, type):
    """
    Convert a List of Sentences to A feature matrix based on the type of data
    :param X: List
    :param feature_matrix: Empty list to Feed in the features
    :param type: train/test
    :return: Feature Matrix
    """
    for row in X:
        feature = []
        if type != 'test':
            refi = row.split('|')[-1]
        else:
            refi = row
        words = refi.split()
        avg = sum(len(word) for word in words) / len(words)
        if avg > 5.0:
            feature.append(True)
        else:
            feature.append(False)
        avg_consonants_in_row = get_con_count(words)
        if avg_consonants_in_row > 4.3:
            feature.append(True)
        else:
            feature.append(False)
        if ('the' in refi):
            feature.append(True)
        else:
            feature.append(False)
        is_common_dutch = False
        for word in refi:
            if word in DUTCH_COMMON_WORDS:
                is_common_dutch = True
        if is_common_dutch == True:
            feature.append(True)
        else:
            feature.append(False)
        is_common_english = False
        for word in refi:
            if word in ENGLISH_COMMON_WORDS:
                is_common_english = True
        if is_common_english == True:
            feature.append(True)
        else:
            feature.append(False)
        if type != 'test':
            feature.append(row.split('|')[0])
        feature_matrix.append(feature)
    return feature_matrix


def split(column, data):
    """
    Split a dataset based on Column value
    :param column: Column index
    :param data: Dataset
    :return: Return subsets
    """
    right_list = []
    left_list = []
    for elem in data:
        if elem[column] == True:
            right_list.append(elem)
        else:
            left_list.append(elem)
    return right_list, left_list


def calculate_entropy(column, dataset):
    """
    Calculate Entropy of A column in dataset
    :param column: Column Index
    :param dataset: Dataset
    :return:
    """
    unique_values = set([data[column] for data in dataset])
    value_counts = {}
    for i in unique_values:
        value_counts[i] = 0
    for elem in dataset:
        for value in unique_values:
            if elem[column] == value:
                value_counts[value] += 1
    entropy = 0
    for count in value_counts:
        probability = value_counts[count] / len(dataset)
        entropy += probability * math.log(probability, 2)
    return -entropy


def cal_information_gain(col, dataset, target):
    """
    Calculate Information Gain of a Column based on the dataset
    :param col: Column index
    :param dataset:
    :param target:
    :return: IG of a column
    """
    initial_entropy = calculate_entropy(target, dataset)
    right, left = split(col, dataset)
    remainder = 0
    prob_left = len(left) / len(dataset)
    prob_right = len(right) / len(dataset)
    remainder += prob_left * calculate_entropy(target, left)
    remainder += prob_right * calculate_entropy(target, right)

    return initial_entropy - remainder


def best_attribute_for_split(columns, dataset):
    """
    Find the best column with the Max IG
    :param columns: column List
    :param dataset:
    :return: Column index and IG
    """
    ig = {}
    for col in columns:
        information_gain = cal_information_gain(col, dataset, -1)
        ig[col] = information_gain
    best_col = max(ig, key=ig.get)
    return best_col, ig[best_col]


def predict(predictions: dict):
    """
    Based on the counts of label , return the max value
    :param predictions:
    :return: Label with max count
    """
    best_count = 0
    result = []
    for label in predictions:
        if predictions[label] > best_count:
            result.clear()
            result.append(label)
            best_count = predictions[label]
        elif predictions[label] == best_count:
            result.append(label)
    return result[0]


def get_label_count(matrix):
    """
    Get the Count of labels in a list
    :param matrix:
    :return: label count dictionary
    """
    count = {}
    for row in matrix:
        label = row[-1]
        if label not in count:
            count[label] = 1
        else:
            count[label] += 1
    return count


class Node:
    __slots__ = ['column', 'predictions', 'true_branch', 'false_branch']

    def __init__(self, column, matrix, true_branch, false_branch):
        self.column = column
        self.predictions = get_label_count(matrix)
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __str__(self):
        return f'{str(self.column)} {str(self.predictions)}'


class Leaf:
    __slots__ = ['predictions']

    def __init__(self, matrix):
        self.predictions = get_label_count(matrix)

    def __str__(self):
        return str(self.predictions)


def buildTree(dataset):
    """
    Recursive method to Build Decision Tree
    :param dataset:
    :return: Decision tree
    """
    column, info_gain = best_attribute_for_split([i for i in range(len(dataset[0]) - 1)], dataset)
    if info_gain == 0:
        return Leaf(dataset)
    right, left = split(column, dataset)
    left_branch = buildTree(left)
    right_branch = buildTree(right)
    return Node(column, dataset, right_branch, left_branch)


def classification(tuple, node):
    """
    Predict a target for tuple
    :param tuple: row
    :param node: tree model
    :return: Predict method for getting the target
    """
    if isinstance(node, Leaf):
        return predict(node.predictions)
    if tuple[node.column] == True:
        return classification(tuple, node.true_branch)
    else:
        return classification(tuple, node.false_branch)

def train(dataset, weights):
    """
    Method to learn the best column of the dataset after choosing the new dataset
    :param dataset:
    :param weights:
    :return: column and new dataset
    """
    weight_limits = []
    curr_w = 0
    for w in weights:
        curr_w += w
        weight_limits.append(curr_w)
    new_dataset = []
    for i in range(len(dataset)):
        rand_weight = random.uniform(0, 1)
        for j in range(len(dataset)):
            if rand_weight < weight_limits[j]:
                new_dataset.append(dataset[j])
                break
    column, ig = best_attribute_for_split([i for i in range(len(new_dataset[0]) - 1)], new_dataset)
    return new_dataset, column


def normalize_weights(weights):
    """
    Normalize weights array based on updated values
    :param weights:
    :return: normalized array
    """
    total = sum(weights)
    weight_array = []
    for weight in weights:
        weight_array.append(weight / total)
    return weight_array


def adaBoost(feature_matrix):
    """
    Ada Boost Algorithm which
    :param feature_matrix:
    :return: List of Hypothesis and Z values
    """
    no_of_features = len(feature_matrix[0]) - 1
    result = [feature_matrix[i][-1] == 'en' for i in range(len(feature_matrix))]
    w = [(1 / len(feature_matrix)) for i in range(len(feature_matrix))]
    z = [0 for _ in range(no_of_features)]
    h = [None for _ in range(no_of_features)]
    updated_dataset = feature_matrix
    for k in range(no_of_features):
        updated_dataset1, h[k] = train(updated_dataset, w)
        error = 0
        for j in range(len(updated_dataset1)):
            if updated_dataset1[j][k] is not result[j]:
                error += w[j]
        for j in range(len(updated_dataset1)):
            if updated_dataset1[j][k] is result[j]:
                w[j] = w[j] * (error / (1 - error))
        w = normalize_weights(w)
        if error == 0:
            z[k] = float('inf')
        elif error == 1:
            z[k] = 0
        else:
            z[k] = math.log((1 - error) / error)
    hypothesis = [(h[k], z[k]) for k in range(no_of_features)]
    return hypothesis

def ada_classify(tuple, tree_list):
    """
    Classify a tuple based on the Hypothesis List
    :param tuple: row
    :param tree_list: Hypothesis List
    :return: Target Label
    """
    nl = 0
    en = 0
    for tree in tree_list:
        x = tuple[tree[0]]
        if x:
            en += tree[1]
        else:
            nl += tree[1]
    if nl < en:
        return "Predicted Label is: nl"
    else:
        return "Predicted Label is: en"


def classify(data, node):
    """
    Find whether to predict using ada or decision tree
    :param data:
    :param node:
    :return: selected classification function
    """
    if isinstance(node, list):
        return ada_classify(data, node)
    else:
        return classification(data, node)


def main():
    """
    Main method to retreive all the system arguments and print the result
    """
    operation = sys.argv[1]
    hypothesis = None
    if operation == "train":
        learning_type = sys.argv[4]
        input_file = sys.argv[2]
        hypothesis_out_file = sys.argv[3]
        input = get_input(input_file)
        decision_matrix = get_features(input, [], 'train')
        if learning_type == 'dt':
            hypothesis = buildTree(decision_matrix)
        elif learning_type == 'ada':
            hypothesis = adaBoost(decision_matrix)
        pickle.dump(hypothesis, open(hypothesis_out_file, 'wb'))
        print("Training Completed")
    elif operation == "predict":
        hypothesis_out_file = sys.argv[2]
        input_file = get_input(sys.argv[3])
        node = pickle.load(open(hypothesis_out_file, 'rb'))
        test_matrix = get_features(input_file, [], 'test')
        for data in test_matrix:
            x = classify(data, node)
            print(x)


if __name__ == "__main__":
    main()
