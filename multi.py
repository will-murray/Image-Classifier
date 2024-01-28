import mnist_reader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def initialize_data(kind):
    X,y = mnist_reader.load_mnist(path= "data/fashion" , kind = kind)
    #reshape (flatten the 28x28 images into feature vectors
    X = X.reshape(X.shape[0], 784)
    y = y.reshape(y.shape[0], )
    #normalize the values
    X = X/255    

    return X,y


def fetch_data(size = 10000):
    X_train, y_train = initialize_data('train')
    X_test, y_test = initialize_data('t10k')

    return X_train[:size, ], y_train[:size, ] , X_test[:int(size/6), ], y_test[:int(size/6), ] 


X,y,X_test,y_test = fetch_data(size = 3000)

model = OneVsRestClassifier(estimator=SVC()).fit(X,y)
y_hat = model.predict(X_test)

confusion_matrix = np.zeros((10,10))
for a,b in zip(y_test, y_hat):
    confusion_matrix[a,b] += 1


plt.imshow(confusion_matrix,vmax=np.max(confusion_matrix), vmin= np.min(confusion_matrix))
plt.legend()
plt.show()