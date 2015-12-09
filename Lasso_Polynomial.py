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
from matplotlib.legend_handler import HandlerLine2D
########################training######################



X= []
y =[]
X_test = []
y_test =[]
X_val = []
y_val = []
X_val2 = []
y_val2 = []
X_val3 = []
y_val3 = []
X_val4 = []
y_val4 = []
X_train = []
y_train = []
X_test = []
y_test = []

with open('Trainfeature_nm_less.txt') as f:
	x = f.read().splitlines()




for line in x[160000:180000]: #training 
    temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
    storeOpen = temp_list[3]
    if storeOpen != 0:
     X.append(temp_list)
     y.append(int(line.split()[-1]))
     
for line in x[900000:920000]: #training 
    temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
    storeOpen = temp_list[3]
    if storeOpen != 0:
     X.append(temp_list)
     y.append(int(line.split()[-1]))


# print '------------'

for line in x[4000:6209]: # Big Validation outside train
    temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
    storeOpen = temp_list[3]
    if storeOpen != 0:
     X_val.append(temp_list)
     y_val.append(int(line.split()[-1]))

X_train = X
y_train = y

for line in x[120000:122209]: # Big Validation outside train
    temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
    storeOpen = temp_list[3]
    if storeOpen != 0:
     X_val2.append(temp_list)
     y_val2.append(int(line.split()[-1]))

for line in x[470000:472209]: # Big Validation outside train
    temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
    storeOpen = temp_list[3]
    if storeOpen != 0:
     X_val3.append(temp_list)
     y_val3.append(int(line.split()[-1]))

for line in x[1012000:1014209]: # Big Validation outside train
    temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
    storeOpen = temp_list[3]
    if storeOpen != 0:
     X_val4.append(temp_list)
     y_val4.append(int(line.split()[-1]))

# for line in x[120000:140000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))
     
# for line in x[800000:820000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))


# # print '------------'

# for line in x[3000:5209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val.append(temp_list)
#      y_val.append(int(line.split()[-1]))

# X_train = X
# y_train = y

# for line in x[240000:242209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val2.append(temp_list)
#      y_val2.append(int(line.split()[-1]))

# for line in x[520000:522209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val3.append(temp_list)
#      y_val3.append(int(line.split()[-1]))

# for line in x[981000:983209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val4.append(temp_list)
#      y_val4.append(int(line.split()[-1]))




# for line in x[20000:40000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))
     
# for line in x[970000:990000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))


# # print '------------'

# for line in x[10000:12209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val.append(temp_list)
#      y_val.append(int(line.split()[-1]))


# for line in x[30000:32209]: # big Validation inside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_train.append(temp_list)
#      y_train.append(int(line.split()[-1]))

# for line in x[200000:202209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val2.append(temp_list)
#      y_val2.append(int(line.split()[-1]))

# for line in x[300000:302209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val3.append(temp_list)
#      y_val3.append(int(line.split()[-1]))

# for line in x[1015000:1017209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val4.append(temp_list)
#      y_val4.append(int(line.split()[-1]))







# for line in x[60000:80000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))
     
# for line in x[980000:1000000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))
     


# # print '------------'

# for line in x[20000:22209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val.append(temp_list)
#      y_val.append(int(line.split()[-1]))


# for line in x[70000:72209]: # big Validation inside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_train.append(temp_list)
#      y_train.append(int(line.split()[-1]))

# for line in x[100000:102209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val2.append(temp_list)
#      y_val2.append(int(line.split()[-1]))

# for line in x[350000:352209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val3.append(temp_list)
#      y_val3.append(int(line.split()[-1]))

# for line in x[1012000:1014209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val4.append(temp_list)
#      y_val4.append(int(line.split()[-1]))



# for line in x[80000:100000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))
     

# for line in x[720000:740000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))

# # print '------------'

# for line in x[0:2209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val.append(temp_list)
#      y_val.append(int(line.split()[-1]))


# for line in x[90000:92209]: # big Validation inside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_train.append(temp_list)
#      y_train.append(int(line.split()[-1]))

# for line in x[57000:59209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val2.append(temp_list)
#      y_val2.append(int(line.split()[-1]))

# for line in x[210000:212209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val3.append(temp_list)
#      y_val3.append(int(line.split()[-1]))

# for line in x[1002000:1004209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val4.append(temp_list)
#      y_val4.append(int(line.split()[-1]))



# for line in x[100000:120000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))
     
# for line in x[50000:70000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))
     


# # print '------------'

# for line in x[3000:5209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val.append(temp_list)
#      y_val.append(int(line.split()[-1]))


# for line in x[113000:115209]: # big Validation inside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_train.append(temp_list)
#      y_train.append(int(line.split()[-1]))

# for line in x[8100:10309]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val2.append(temp_list)
#      y_val2.append(int(line.split()[-1]))

# for line in x[156000:158209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val3.append(temp_list)
#      y_val3.append(int(line.split()[-1]))

# for line in x[1006000:1008209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val4.append(temp_list)
#      y_val4.append(int(line.split()[-1]))


# for line in x[200000:220000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))
     
# for line in x[500000:520000]: #training 
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X.append(temp_list)
#      y.append(int(line.split()[-1]))


# # print '------------'

# for line in x[112345:114554]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val.append(temp_list)
#      y_val.append(int(line.split()[-1]))


# for line in x[200100:202309]: # big Validation inside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_train.append(temp_list)
#      y_train.append(int(line.split()[-1]))

# for line in x[320000:322209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val2.append(temp_list)
#      y_val2.append(int(line.split()[-1]))

# for line in x[400000:402209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val3.append(temp_list)
#      y_val3.append(int(line.split()[-1]))

# for line in x[906000:908209]: # Big Validation outside train
#     temp_list = map(float, line.split()[0:-1])#[int(x) for x in line.split() if x != '\n' and x != '']
#     storeOpen = temp_list[3]
#     if storeOpen != 0:
#      X_val4.append(temp_list)
#      y_val4.append(int(line.split()[-1]))


x = []
error= []
degs=[]
var = []

y_vars = []
score=[]
error_train = []
var_train = []
score_train = []

error_2 = []
var_2 = []
score_2 = []

error_3 = []
var_3 = []
score_3 = []

error_4 = []
var_4 = []
score_4 = []
ss=[] # score of training set
for deg in range(1,5):
    print 'aa'
    poly = PolynomialFeatures(degree= deg)
    print 'bb'
    X_ = poly.fit_transform(X)
    print 'cc'
    X_val_ = poly.fit_transform(X_val)
    X_train_ = poly.fit_transform(X_train)
    X_val2_ = poly.fit_transform(X_val2)
    X_val3_ = poly.fit_transform(X_val3)
    X_val4_ = poly.fit_transform(X_val4)


    for a in range(0,11):
        it = 0.001 * a
        print it
        

        print 'dd'
        clf = linear_model.Lasso(alpha = it)
        print 'ee'
        clf.fit(X_,y)

        print 'ff'
        weights =  clf.coef_
        print 'gg'
        print 'weights:'
        print weights

        print np.array(weights).shape
        # weights = weights.tolist()

        y_true = y_val
        y_true_train = y_train

        y_pred = clf.predict(X_val_)
        y_pred_train = clf.predict(X_train_)

        y_true2 = y_val2
        y_pred2 = clf.predict(X_val2_)

        y_true3 = y_val3
        y_pred3 = clf.predict(X_val3_)

        y_true4 = y_val4
        y_pred4 = clf.predict(X_val4_)


        e =  mean_squared_error(y_true, y_pred)
        v  = explained_variance_score(y_true, y_pred) 
        s = clf.score(X_val_,y_true) 


        eT =  mean_squared_error(y_true_train, y_pred_train)
        vT = explained_variance_score(y_true_train, y_pred_train) 
        sT = clf.score(X_train_,y_true_train) 

        e2 =  mean_squared_error(y_true2, y_pred2)
        v2 = explained_variance_score(y_true2, y_pred2) 
        s2 = clf.score(X_val2_,y_true2)

        e3 =  mean_squared_error(y_true3, y_pred3)
        v3 = explained_variance_score(y_true3, y_pred3) 
        s3 = clf.score(X_val3_,y_true3)  

        e4 =  mean_squared_error(y_true4, y_pred4)
        v4 = explained_variance_score(y_true4, y_pred4) 
        s4 = clf.score(X_val4_,y_true4)  



        STT = clf.score(X_,y) 


        print 'jj'
        print e
        print v
        print s

        degs.append(deg)
        x.append(it)

        error.append(e)
        var.append(v)
        score.append(s)

        error_train.append(eT)
        var_train.append(vT)
        score_train.append(sT)

        error_2.append(e2)
        var_2.append(v2)
        score_2.append(s2)

        error_3.append(e3)
        var_3.append(v3)
        score_3.append(s3)

        error_4.append(e4)
        var_4.append(v4)
        score_4.append(s4)

        ss.append(STT)


print("Best error in validation set 0 is %0.4f " % np.amin(np.array(error_train)))
print("Best error index in validation set 0 is %0.4f " % np.argmin(np.array(error_train)))
print("Best variance score in validation set 0 is %0.4f " % np.amax(np.array(var_train)))
print("Best varianace score index in validation set 0 is %0.4f " % np.argmax(np.array(var_train)))
print("Best score in validation set 0 is %0.4f " % np.amax(np.array(score_train)))
print("Best score index in validation set 0 is %0.4f " % np.argmax(np.array(score_train)))
print("========================================")

print("Best error in validation set 1 is %0.4f " % np.amin(np.array(error)))
print("Best error index in validation set 1 is %0.4f " % np.argmin(np.array(error)))
print("Best variance score in validation set 1 is %0.4f " % np.amax(np.array(var)))
print("Best varianace score index in validation set 1 is %0.4f " % np.argmax(np.array(var)))
print("Best score in validation set 1 is %0.4f " % np.amax(np.array(score)))
print("Best score index in validation set 1 is %0.4f " % np.argmax(np.array(score)))
print("========================================")

print("Best error in validation set 2 is %0.4f " % np.amin(np.array(error_2)))
print("Best error index in validation set 2 is %0.4f " % np.argmin(np.array(error_2)))
print("Best variance score in validation set 2 is %0.4f " % np.amax(np.array(var_2)))
print("Best varianace score index in validation set 2 is %0.4f " % np.argmax(np.array(var_2)))
print("Best score in validation set 2 is %0.4f " % np.amax(np.array(score_2)))
print("Best score index in validation set 2 is %0.4f " % np.argmax(np.array(score_2)))
print("========================================")

print("Best error in validation set 3 is %0.4f " % np.amin(np.array(error_3)))
print("Best error index in validation set 3 is %0.4f " % np.argmin(np.array(error_3)))
print("Best variance score in validation set 3 is %0.4f " % np.amax(np.array(var_3)))
print("Best varianace score index in validation set 3 is %0.4f " % np.argmax(np.array(var_3)))
print("Best score in validation set 3 is %0.4f " % np.amax(np.array(score_3)))
print("Best score index in validation set 3 is %0.4f " % np.argmax(np.array(score_3)))
print("========================================")

print("Best error in validation set 4 is %0.4f " % np.amin(np.array(error_4)))
print("Best error index in validation set 4 is %0.4f " % np.argmin(np.array(error_4)))
print("Best variance score in validation set 4 is %0.4f " % np.amax(np.array(var_4)))
print("Best varianace score index in validation set 4 is %0.4f " % np.argmax(np.array(var_4)))
print("Best score in validation set 4 is %0.4f " % np.amax(np.array(score_4)))
print("Best score index in validation set 4 is %0.4f " % np.argmax(np.array(score_4)))



xi, yi = np.linspace(np.array(x).min(), np.array(x).max(), 100), np.linspace(np.array(degs).min(), np.array(degs).max(), 100)
xi, yi = np.meshgrid(xi, yi)


plt.figure(1)
rbf2 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(var), function='linear')
zi = rbf2(xi, yi)
plt.imshow(zi, vmin=np.array(var).min(), vmax=np.array(var).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 1)Vairance score of Polynomial degree vs. alpha')


plt.figure(2)
rbf = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(error), function='linear')
zi = rbf(xi, yi)
plt.imshow(zi, vmin=np.array(error).min(), vmax=np.array(error).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
# plt.colorbar()
plt.title('(Validation 1)MSE of Polynomial degree vs. alpha')


plt.figure(3)
rbf3 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(score), function='linear')
zi = rbf3(xi, yi)
plt.imshow(zi, vmin=np.array(score).min(), vmax=np.array(score).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 1)score of Polynomial degree vs. alpha')

plt.figure(4)
rbf2 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(var_train), function='linear')
zi = rbf2(xi, yi)
plt.imshow(zi, vmin=np.array(var_train).min(), vmax=np.array(var_train).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 0)Vairance score of Polynomial degree vs. alpha')


plt.figure(5)
rbf = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(error_train), function='linear')
zi = rbf(xi, yi)
plt.imshow(zi, vmin=np.array(error_train).min(), vmax=np.array(error_train).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
# plt.colorbar()
plt.title('(Validation 0)MSE of Polynomial degree vs. alpha')


plt.figure(6)
rbf3 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(score_train), function='linear')
zi = rbf3(xi, yi)
plt.imshow(zi, vmin=np.array(score_train).min(), vmax=np.array(score_train).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 0)score of Polynomial degree vs. alpha')




plt.figure(7)
rbf2 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(var_2), function='linear')
zi = rbf2(xi, yi)
plt.imshow(zi, vmin=np.array(var_2).min(), vmax=np.array(var_2).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 2)Vairance score of Polynomial degree vs. alpha')


plt.figure(8)
rbf = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(error_2), function='linear')
zi = rbf(xi, yi)
plt.imshow(zi, vmin=np.array(error_2).min(), vmax=np.array(error_2).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
# plt.colorbar()
plt.title('(Validation 2)MSE of Polynomial degree vs. alpha')


plt.figure(9)
rbf3 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(score_2), function='linear')
zi = rbf3(xi, yi)
plt.imshow(zi, vmin=np.array(score_2).min(), vmax=np.array(score_2).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 2)score of Polynomial degree vs. alpha')

plt.figure(10)
rbf2 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(var_3), function='linear')
zi = rbf2(xi, yi)
plt.imshow(zi, vmin=np.array(var_3).min(), vmax=np.array(var_3).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 3)Vairance score of Polynomial degree vs. alpha')


plt.figure(11)
rbf = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(error_3), function='linear')
zi = rbf(xi, yi)
plt.imshow(zi, vmin=np.array(error_3).min(), vmax=np.array(error_3).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
# plt.colorbar()
plt.title('(Validation 3)MSE of Polynomial degree vs. alpha')


plt.figure(12)
rbf3 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(score_3), function='linear')
zi = rbf3(xi, yi)
plt.imshow(zi, vmin=np.array(score_3).min(), vmax=np.array(score_3).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 3)score of Polynomial degree vs. alpha')


plt.figure(13)
rbf2 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(var_4), function='linear')
zi = rbf2(xi, yi)
plt.imshow(zi, vmin=np.array(var_4).min(), vmax=np.array(var_4).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 4)Vairance score of Polynomial degree vs. alpha')


plt.figure(14)
rbf = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(error_4), function='linear')
zi = rbf(xi, yi)
plt.imshow(zi, vmin=np.array(error_4).min(), vmax=np.array(error_4).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
# plt.colorbar()
plt.title('(Validation 4)MSE of Polynomial degree vs. alpha')


plt.figure(15)
rbf3 = scipy.interpolate.Rbf(np.array(x), np.array(degs), np.array(score_4), function='linear')
zi = rbf3(xi, yi)
plt.imshow(zi, vmin=np.array(score_4).min(), vmax=np.array(score_4).max(), origin='lower',
           extent=[np.array(x).min(), np.array(x).max(), np.array(degs).min(), np.array(degs).max()])
plt.colorbar()
plt.title('(Validation 4)score of Polynomial degree vs. alpha')



plt.show()




