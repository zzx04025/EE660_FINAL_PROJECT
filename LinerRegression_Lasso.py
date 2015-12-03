def column(matrix, i):
    return [row[i] for row in matrix]


# Generate data
def f(x):
    x = x.ravel()

    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

from sklearn import linear_model
from sklearn import cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import scipy.interpolate
########################training######################



X= []
y =[]
X_test = []
y_test =[]
X_val = []
y_val = []
with open('Trainfeature_nm_less.txt') as f:
	x = f.read().splitlines()

features2 = []
for line in x[0:20000]:
    temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
    storeOpen = temp_list[3]
    if storeOpen != 0:
     X.append(temp_list)
     y.append(int(line.split()[-1]))
     features2.append(temp_list[14])

# print features2
print np.var(np.array(features2))
print '------------'

for line in x[1016001:1017209]:
    temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
    storeOpen = temp_list[3]
    if storeOpen != 0:
     X_val.append(temp_list)
     y_val.append(int(line.split()[-1]))


# for line in x[500000:600000]:
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_test.append(temp_list)
#      y_test.append(int(line.split()[-1]))


# print 'features:'
# print X
# print 'sales:'
# print y
x = []
error= []
degs=[]
var = []
noise = []
bias = []
y_vars = []
shape = []
for deg in range(1,5):
    print 'aa'
    poly = PolynomialFeatures(degree= deg)
    print 'bb'
    X_ = poly.fit_transform(X)
    print 'cc'
    X_val_ = poly.fit_transform(X_val)
    for a in range(1,60):
        it = 0.05 * a
        print it
        

        # clf = linear_model.Ridge (alpha = .1)
        print 'dd'
        clf = linear_model.Lasso(alpha = it)
        print 'ee'
        clf.fit(X_,y)

        print 'ff'
        weights =  clf.coef_
        print 'gg'
        print 'weights:'
        print weights

        # weights = weights.tolist()

        y_true = y_val

        y_pred = clf.predict(X_val_)
        print 'hh'
        # print y_true
        # print y_pred

        e =  mean_squared_error(y_true, y_pred)
        print 'ii'
        v  = explained_variance_score(y_true, y_pred)  
        print 'jj'
        print e
        print v
        # print np.var(np.array(y_true), axis=1)
        # print (f(np.array(X_val_)) - np.mean(np.array(y_pred), axis=1)) ** 2
        # print np.var(np.array(y_pred), axis=1)
        degs.append(deg)
        x.append(it)
        error.append(e)
        var.append(v)
        # noise.append(np.var(np.array(y_true), axis=1))
        # bias.append((f(np.array(X_val_)) - np.mean(np.array(y_pred), axis=1)) ** 2)
        # y_vars.append(np.var(np.array(y_pred), axis=1))
        # scores = cross_validation.cross_val_score(clf, X_test, y_test, cv=5)
        # print scores

# print x
# plt.figure(1)
# plt.plot(x, error)
# plt.title('MSE for each polynomial degree')
# # plt.show()
# plt.figure(2)
# plt.plot(x,var)
# plt.title('Variance score for each polynomial degree')

xi, yi = np.linspace(np.array(x).min(), np.array(x).max(), 100), np.linspace(np.array(degs).min(), np.array(degs).max(), 100)
xi, yi = np.meshgrid(xi, yi)

# rbf = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(var), function='linear')
# zi = rbf(xi, yi)


# plt.imshow(zi, vmin=np.array(var).min(), vmax=np.array(var).max(), origin='lower',
#            extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
# plt.scatter(np.array(x), np.array(degs), c=np.array(var))
rbf = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(error), function='linear')
zi = rbf(xi, yi)
plt.imshow(zi, vmin=np.array(error).min(), vmax=np.array(error).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.scatter(np.array(x), np.array(degs), c=np.array(error))
plt.colorbar()
plt.title('MSE of Polynomial degree vs. alpha')
plt.show()

# scores = cross_validation.cross_val_score(clf, X_test, y_test, cv=5)
# print scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print np.array(weights).shape
#################validation##########################
# X_val= []
# y_val =[]

# for line in x[16:]:
#     temp_list = map(int, line.split()[1:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     X_val.append(temp_list)
#     y_val.append(int(line.split()[-1]))

# # print X_val
# # print y_val

# scores = cross_validation.cross_val_score(clf,X_val,y_val)
# print scores

##################testing############################


