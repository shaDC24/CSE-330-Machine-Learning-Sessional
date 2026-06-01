import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# TODO: Load the dataset
from sklearn.datasets import load_breast_cancer,load_iris
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
# df=df.iloc[:,1:] ..to drop id..but here no id
print(df.head())
print(data.target)
data =load_iris()
df=pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
print(data.target)
df['target'] = data.target

'''
# jodi target ta string e ache ... tahole amra label encoder diye take number e convert korte parbo

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
df['Species']=encoder.fit_transform(df['Species'])
'''

print(df.size)
df=df.sample(100)# erokom korle randomly divide hoe jabe
df_train=df.iloc[:60,:].sample(10)#60 ta random row nie nilam
df_val=df.iloc[60:80,:].sample(5)
df_test=df.iloc[80:,:].sample(5)


X=df_val.iloc[:,0:2]
y=df_val.iloc[:,-1]

X_test=df_val.iloc[:,0:2].values
Y_test=df_val.iloc[:,-1].values


# case-1 Bagging


df_bag=df_train.sample(8,replace=True) #80% row nibo total datarows er ... replace =True mane holo with replacement kaj korbo

from sklearn.tree import DecisionTreeClassifier,plot_tree
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score


dt_bag1=DecisionTreeClassifier()#ekhane depth maximum... so eta LBHV automatically hoe jabe
def evaluate(clf,X,y):
    clf.fit(X,y)
    plot_tree(clf)
    plt.show()
    plot_decision_regions(X.values,y.values,clf=clf,legend=2)
    y_pred=clf.predict(X_test)
    print(accuracy_score(Y_test,y_pred))

evaluate(dt_bag1,X,y)    


