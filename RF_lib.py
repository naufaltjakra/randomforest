import pandas as pd
import numpy as np
import time

import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections


# Inisiasi start timer
start_time = time.time()


#Import dataset
dataset = pd.read_csv('phishing_dataset_small.csv')

# dataset.head()

X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 16].values


data_feature_names = [ 'IP', 'URL Length', 'A Symbol', 'Prefix Suffix', 'Sub Domain', 'HTTPS Token',
                    'Request URL', 'URL Anchor', 'SFH', 'Abnormal URL', 'Redirect', 'On Mouseover',
                    'Pop Up Window', 'Age of Domain', 'DNS Record', 'Web Traffic' ]


#Train Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=7, random_state=0, criterion='entropy', bootstrap=True, max_depth=4)
classifier.fit(X, y)
y_pred = classifier.predict(X_test)


##Try To Read The Tree
#Export Tree in file.dot for every tree
# from sklearn import tree
# i_tree = 0
# for tree_in_forest in classifier.estimators_:
#     with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
#         my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
#     i_tree = i_tree + 1


# Visualize data
dot_data = tree.export_graphviz(classifier.estimators_[2],
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('gray', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')



#Accuracy in K-Fold
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)


#Display accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#Print the accuracy in every folds
print("Folds :")
print(accuracy*100)
print('Mean Accuracy: %.3f%%' % (sum(accuracy)/float(len(accuracy))*100))
print('Maximum Accuracy: %.3f%%' % max(accuracy*100))

print(" ")
print("Feature Importance :")
print(classifier.feature_importances_)

# Display render time
elapsed_time = time.time() - start_time
time.strftime("Render Time : %H:%M:%S", time.gmtime(elapsed_time))