from typing_extensions import ParamSpec
import streamlit as st
from sklearn import datasets # sklearn has some common inbuilt dataset 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.title ("Machine Learning App")

st.write ("""
# Explore different classifiers

Which one is the best

""")

#create a diagloge box that contains a set of data
# dataset_name = st.selectbox ("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))

#move to side bar
dataset_name = st.sidebar.selectbox ("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))

#st.write(dataset_name)

#create another select box for classifier
classifier_name = st.sidebar.selectbox ("Select Classifier", ("KNN", "SVM", "Random Forest"))

# define a function to pull these dataset from sklearn

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris() #included in scikilean
    elif dataset_name == "Breast Cancer":
        data =datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

#lets call the function

X, y = get_dataset (dataset_name)

#wriite from output to test the dataset
st.write ("shape of dataset", X.shape)
st.write ("number of classes", len(np.unique(y)))

#Add different parameter that we can modify for our classifier
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)  #K is an important parameter in KNN here we are starting from 1 to 15
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0) # check SVM documentation
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15) #no of depth for tree
        n_estimators = st.sidebar.slider("n_estimators", 1, 100) #no of tree
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params 

params = add_parameter_ui(classifier_name)

#Create the actual classifier
def get_classifier (clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C= params["C"])
    else:
        clf = RandomForestClassifier(n_estimators = params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

#Now we can do our classification
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score (y_test, y_pred)

#write to app
st.write (f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")


#Plotting the graph
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected [:, 0]
x2 = X_projected [:, 1]

fig = plt.figure()
plt.scatter (x1, x2, c=y, alpha= 0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)