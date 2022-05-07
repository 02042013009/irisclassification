from sklearn import datasets
import numpy as np
import pickle

iris = datasets.load_iris()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,shuffle=True,stratify=iris.target,random_state=5)

from sklearn.tree import DecisionTreeClassifier
dtmodel = DecisionTreeClassifier(criterion='entropy',random_state=1)
dtmodel.fit(x_train,y_train)

pickle.dump(dtmodel,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

x_test = np.array([5.1,3.5,1.4,2.0])
x_test = x_test.reshape(1,-1)

print(model.predict(x_test))