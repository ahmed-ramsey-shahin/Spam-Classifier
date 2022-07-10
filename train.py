# -*- coding: utf-8 -*-
from sklearn import svm
import numpy as np
import pickle

# Load the dataset
# The data is already shuffled
dataset = np.load('dataset.npy')

# Split the data into train, cross validation, and test
trainset_size = 7738 # 60% of the data
crossvalidation_size = 2579 # 20% of the data
test_size = 2579 # 20% of the data

X_train = dataset[0:trainset_size]
y_train = X_train[:, -1]
X_train = X_train[:, :-1]

X_cv = dataset[trainset_size:(trainset_size + crossvalidation_size)]
y_cv = X_cv[:, -1]
X_cv = X_cv[:, :-1]

X_test = dataset[(trainset_size + crossvalidation_size):]
y_test = X_test[:, -1]
X_test = X_test[:, :-1]

# Choose the best value for C, and gamma then train the model
C = 0
gamma = 0
values = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
cv_err = 1
clf = None

counter = 1
for C_temp in values:
    for gamma_temp in values:
        clf_temp = svm.SVC(kernel='rbf', gamma=1/(2 * (gamma_temp)**2), C=C_temp)
        clf_temp.fit(X_train, y_train)
        predictions_cv = clf_temp.predict(X_cv)
        err = np.mean(predictions_cv != y_cv)
        print('{} - For C={}, and gamma={}, the error: {}'.format(counter, C_temp, gamma_temp, err * 100))
        counter += 1
        if cv_err > err:
            cv_err = err
            C = C_temp
            gamma = gamma_temp
            clf = clf_temp

predictions_test = clf.predict(X_test)
global_err = np.mean(predictions_test != y_test)

print('Cross validation set error: {}'.format(cv_err * 100))
print('Global error: {}'.format(global_err * 100))

# Save the classifier to model
file = open('classifier.obj', 'wb')
pickle.dump(clf, file)
file.close()
