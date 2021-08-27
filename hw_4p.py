import math


def get_input(file_name):
    try:
        data1=[]
        with open(file_name) as f:
            data=f.readlines()
            for line in data:
                data1.append(line.replace('\n','').split(' '))

    except IOError as e:
        print(e)
    return data1



data=get_input('input.txt')
target=[]
for i in range(len(data)):
    target.append(data[i][-1])

col_dict={}
counter=0
for j in range(len(data[0])):
    column=[]

    for i in range(len(data)):
        column.append(data[i][j])
    col_dict[counter]=column
    counter+=1

def cal_entropy(column):
    a=0
    b=0
    if len(column)==0:
        return 0
    value1=column[0]
    for val in column:
        if val==value1:
            a+=1
        else:
            b+=1
    prob_true=a/len(column)
    prob_false=b/len(column)
    entropy=0
    if prob_true>0:
        entropy+=prob_true*math.log(prob_true,2)
    if prob_false>0:
        entropy+=prob_false*math.log(prob_false,2)
    return -entropy

def cal_information_gain(col_dict,col_num,target):
    initial_entropy=cal_entropy(col_dict[target])
    left_split=[]
    right_split = []

    for i in range(len(col_dict[col_num])):
        if col_dict[col_num][i]=="True":
            left_split.append((col_dict[col_num][i],i))
        else:
            right_split.append((col_dict[col_num][i],i))
    remainder=0
    target_left=[]
    for i in range(len(left_split)):
        target_left.append(col_dict[target][left_split[i][1]])
    target_right=[]
    for i in range(len(right_split)):
        target_right.append(col_dict[target][right_split[i][1]])
    prob_left=(len(left_split)/len(col_dict[col_num]))
    remainder+=prob_left*cal_entropy(target_left)
    prob_right=(len(right_split)/len(col_dict[col_num]))
    remainder+=prob_right*cal_entropy(target_right)

    return initial_entropy-remainder
def best_attribute_for_split(columns,col_dict):
    ig={}
    for col in columns:
        information_gain=cal_information_gain(col_dict,col,8)
        ig[col]=information_gain
    print(ig)
    return max(ig,key=ig.get)
def recursive_split(columns,col_dict):
    root=best_attribute_for_split(columns,col_dict)
    first_val=col_dict[root][0]
    left_split=[]
    right_split=[]
    for i in range(len(col_dict[root])):
        if col_dict[root][i]==first_val:
            left_split.append((col_dict[root][i],i))
        else:
            right_split.append((col_dict[root][i],i))
    new_dict_left={}
    new_dict_right = {}
    for i in col_dict:
        left=[]
        right=[]
        for j in left_split:
            left.append(col_dict[i][j[1]])
        for k in right_split:
            right.append(col_dict[i][k[1]])
        new_dict_left[i]=left
        new_dict_right[i]=right
    left_child_column=best_attribute_for_split([0,1,2,3,4,5,6,7],new_dict_left)
    right_child_column=best_attribute_for_split([0, 1, 2, 3, 4, 5, 6, 7], new_dict_right)
    return (root,(left_child_column,new_dict_left),(right_child_column,new_dict_right))

best_columns=recursive_split([0,1,2,3,4,5,6,7],col_dict)
print('for depth 1 best column for split:'+str(best_columns[0]+1))
print('for depth 2 best columns for split:'+str(best_columns[1][0]+1)+' and '+str(best_columns[2][0]+1))
data_1=[]
with open('input.txt') as f:
    # data_1=f.readlines().split()
    for i in f:
        data_1.append(i.split())

# count_b=0
# count_a=0
# print(best_columns[0])

def get_counts(data,best_columns):
    counts = {"True;True": {'A': 0, 'B': 0}, "True;False": {'A': 0, 'B': 0}, "False;True": {'A': 0, 'B': 0},
              "False;False": {'A': 0, 'B': 0}}
    for i in data:
        if i[best_columns[0]] == 'True':
            if i[best_columns[2][0]] == "True":
                if i[-1] == 'A':
                    counts["True;True"]['A'] += 1
                elif i[-1] == 'B':
                    counts["True;True"]['B'] += 1
                # counts["True;True"]={'A':count_a,'B':count_b}
            elif i[best_columns[2][0]] == "False":
                if i[-1] == 'A':
                    # count_a+=1
                    counts["True;False"]['A'] += 1
                else:
                    counts["True;False"]['B'] += 1
        else:
            if i[best_columns[1][0]] == "True":
                if i[-1] == 'A':
                    counts["False;True"]['A'] += 1
                elif i[-1] == 'B':
                    counts["False;True"]['B'] += 1
                # counts["True;True"]={'A':count_a,'B':count_b}
            elif i[best_columns[1][0]] == "False":
                if i[-1] == 'A':
                    # count_a+=1
                    counts["False;False"]['A'] += 1
                else:
                    counts["False;False"]['B'] += 1
    return counts
count=get_counts(data_1,best_columns)
# print("Counts of each leaf of decision Tree:"+count)

if count['True;True']['A']>count['True;True']['B']:
    print("For column 4 true and column 6 true gives: A"+" with count:"+str(count['True;True']['A']))
else:
    print("For column 4 true and column 6 true gives: B" + " with count: " + str(count['True;True']['B']))

if count['True;False']['A']>count['True;False']['B']:
    print('For column 4 true and column 6 false gives: A with count: '+str(count['True;False']['A']))
else:
    print('For column 4 true and column 6 false gives: B with count: ' + str(count['True;False']['B']))

if count['False;False']['A']>count['False;False']['B']:
    print('For column 4 false and column 5 false gives: A with count: '+str(count['False;False']['A']))
else:
    print('For column 4 false and column 5 false gives: B with count: ' + str(count['False;False']['B']))

if count['False;True']['A']>count['False;True']['B']:
    print('For column 4 false and column 5 true gives: A with count: '+str(count['False;True']['A']))
else:
    print('For column 4 false and column 5 true gives: B with count: ' + str(count['False;True']['B']))

# count_a=0
# count_b=0
# for i in data_1:
#     if i[3]=='False':
#         if i[-1]=='A':
#             count_a+=1
#     if i[3]=='True':
#         if i[-1]=='B':
#             count_b+=1
# print((count_b+count_a)/200)
# print(counts)
# print(cal_entropy(col_dict[2]))