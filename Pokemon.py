import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import sys

color = sys.stdout.shell


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#column=['Name','Type 1','Type 2','Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Stage','Legendary']

#data = pd.DataFrame({'values':['Name','Type 1','Type 2','Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Stage','Legendary'] })

df1=pd.read_csv('pokemon.csv',encoding='unicode_escape',index_col=0)

df2=pd.read_csv('Pokemon1.csv',encoding='unicode_escape',index_col=0)


#DATA preprocessing
df1.drop(['Name','Type 1','Type 2','Generation'],inplace=True,axis=1)
df2.drop(['Name','Type 1','Type 2','Total','Stage'],inplace=True,axis=1)

#splitting
features=['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']
target=['Legendary']

X_test=df1.iloc[:,:6].values


y_test=df1.iloc[:,6].values


X_train=df1.iloc[:,:6].values

y_train=df1.iloc[:,6].values

#Algo

clf=RandomForestClassifier(n_estimators=5,criterion='gini',max_depth=7,random_state=1)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

#save tree as png
from matplotlib import image as pltimg
import pydotplus
from sklearn.tree import export_graphviz
classifier=clf.estimators_[1]
data=export_graphviz(classifier,out_file=None,filled=True)
graph=pydotplus.graph_from_dot_data(data)
graph.write_png('figurepoke.png')

img=pltimg.imread('figurepoke.png')
plt.imshow(img)
plt.show()

