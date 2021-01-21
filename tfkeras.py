import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print('tensorflow version = ', tf.__version__)
print('keras version = ', keras.__version__)


from tensorflow.keras import layers as kl
from tensorflow.keras import models as km
from tensorflow.keras import optimizers as ko


X = np.random.randn(25,2)
y = 3.0*X[:,0] + 5.2*X[:,1] + 13.0 + np.random.randn(25)


model = km.Sequential()
model.add(kl.Dense(1,activation='linear',input_dim=2))
model.compile(optimizer=ko.SGD(lr=0.01),loss='mse')


history = model.fit(X,y,epochs=500,verbose=1)
preds = model.predict(X)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history.history['loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
fig.suptitle('Loss from linear regression with NN');
plt.show()

mse = 1/len(y)* np.sum((y-preds.reshape((25,)))**2)
print('final mean square error = ', mse)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y-preds.reshape((25,)),'o',label='residuals')
ax.plot(np.zeros(len(y)),'k',label='zero')
ax.legend(loc='best')
fig.suptitle('Residuals from linear regression with NN');
plt.show()

parameters = model.get_weights()
slopes = parameters[0]
intercept = parameters[1]


print('slope 1 = ',slopes[0][0])
print('slope 2 = ', slopes[1][0])
print('intercept = ', intercept[0])


model2 = km.Sequential()
model2.add(kl.Input(shape=(2,)))
model2.add(kl.Dense(1,activation='linear'))
model2.compile(optimizer=ko.SGD(lr=0.01),loss='mse')
history2 = model2.fit(X,y,epochs=500,verbose=0)
preds2 = model2.predict(X)
model2.get_weights()


X = np.random.uniform(5,25,size=(100,2))
y_s = -6.0*X[:,0] + 5.2*X[:,1] + 13.0
y_prob = np.exp(y_s)/(1 + np.exp(y_s))
y = np.random.binomial(1,y_prob,100)


print(y)



model3 = km.Sequential()
model3.add(kl.Dense(1,activation='sigmoid',input_dim=2))
model3.compile(optimizer=ko.SGD(),loss='binary_crossentropy')
history3 = model3.fit(X,y,epochs=500,verbose=0)



fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history3.history['loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
fig.suptitle('Loss from logistic regression with NN');
plt.show()


preds3 = model3.predict(X)
preds3_class = model3.predict_classes(X)



model3.get_weights()


from sklearn.metrics import confusion_matrix


confusion_matrix(y,preds3_class)


from sklearn.metrics import roc_curve


fpr, tpr, thresholds = roc_curve(y,preds3_class)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(fpr,tpr,label='ROC')
ax.plot([0,1],[0,1],'k--',label=None)
ax.legend(loc='best')
fig.suptitle('ROC from logistic regression with NN');
plt.show()

from sklearn.metrics import roc_auc_score



roc_auc_score(y,preds3_class)


XOR_X = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
XOR_y = np.array([[0],[1],[1],[0]], "float32")



model4 = km.Sequential()
model4.add(kl.Dense(1,activation='sigmoid',input_dim=2))
model4.compile(optimizer=ko.SGD(lr=0.1),loss='binary_crossentropy',show_accuracy=True)
history4 = model4.fit(XOR_X,XOR_y,epochs=1000,verbose=0)



preds4 = model4.predict(XOR_X)
preds4_class = model4.predict_classes(XOR_X)
confusion_matrix(XOR_y,preds4_class)



model5 = km.Sequential()
model5.add(kl.Dense(8,activation='relu',input_dim=2))
model5.add(kl.Dense(1,activation='sigmoid'))
model5.compile(optimizer=ko.SGD(lr=0.1),loss='binary_crossentropy',show_accuracy=True)
history5 = model5.fit(XOR_X,XOR_y,epochs=1000,verbose=0)


preds5 = model5.predict(XOR_X)
preds5_class = model5.predict_classes(XOR_X)
confusion_matrix(XOR_y,preds5_class)


print(XOR_y)

print(preds4_class)

print(preds5_class)

