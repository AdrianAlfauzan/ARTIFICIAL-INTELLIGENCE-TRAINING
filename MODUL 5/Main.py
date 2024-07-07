import pandas as pd
import numpy as np
import math
from anytree import Node, RenderTree
import json

# Load Data
# Jangan lupa juga pip install openpyxl
df = pd.read_excel('D:\\PRAKTIKUM KECERDASAN BUATAN\\MODUL 5\\play_tennis.xlsx')

#Define ID3 Function
def id3(data,target,attributes):
    # Check for base cases
    unique_targets = np.unique(data[target])
    if len(unique_targets) == 1:
        return unique_targets[0]
    elif len(attributes) == 0:
        return df[target].mode()[0]
    
    # Calculate information gain for each attribute
    entropy_s = entropy(data[target])
    gains = {}
    for attribute in attributes:
        entropy_a = 0
        for value in np.unique(data[attribute]):
            subset = data[data[attribute] == value]
            entropy_a += len(subset)/len(data)*entropy(subset[target])
            gains[attribute] = entropy_s - entropy_a
    
    # Choose attribute with highest information gain
    best_attribute = max(gains, key=gains.get)

    # Create subb - Tree
    tree  = {best_attribute : {}}
    for value in np.unique(data[best_attribute]):
        subset = data[data[best_attribute] == value]
        subtree = id3(subset,target,[i for i in attributes if i != best_attribute])
        tree[best_attribute][value] = subtree

    return tree

#  define entropy function
def entropy(data):
    values,counts = np.unique(data,return_counts=True)
    probs = counts/len(data)
    entropy = sum([-prob*math.log2(prob) for prob in probs])
    return entropy

# Run ID3 algorithm
attributes = ['Outlook','Temperatur','Humidity','Wind']
target = 'PlayTennis'
tree = id3(df,target,attributes)
print(tree)

def build_tree(json_obj, parent=None):
    for k, v in json_obj.items():
        node = Node(k, parent=parent)
        if isinstance(v, dict):
            build_tree(v, parent=node)
        else:
            Node(v, parent=node)

json_obj = json.dumps(tree)
json_obj = json.loads(json_obj)

root_node = Node('Root')
build_tree(json_obj, parent=root_node)
for pre, fill, node in RenderTree(root_node):
    print("%s%s" % (pre, node.name))
