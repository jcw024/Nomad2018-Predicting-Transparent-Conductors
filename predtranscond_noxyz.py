#analyzing base model accuracy without the xyz data for comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import scale, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
import time
warnings.filterwarnings("ignore")
start = time.time()

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

def RMSLE(pred1,pred2,true1,true2):
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    true1 = np.array(true1)
    true2 = np.array(true2)
    error1 = np.square(np.log(pred1 + 1) - np.log(true1 + 1)).mean()**0.5
    error2 = np.square(np.log(pred2 + 1) - np.log(true2 + 1)).mean()**0.5
    return np.sum(error1 + error2)/2

X_train = train.head(2200).drop("id",axis=1)
X_val = train.tail(200).drop("id",axis=1)

y_train_1 = X_train["formation_energy_ev_natom"]
y_train_2 = X_train["bandgap_energy_ev"]
y_val_1 = X_val["formation_energy_ev_natom"]
y_val_2 = X_val["bandgap_energy_ev"]

X_train = X_train.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)
X_val = X_val.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)

scaler = StandardScaler().fit(X_train)

svr1 = SVR(C=10)
svr2 = SVR(C=10)
svr1.fit(scale(X_train),y_train_1)
svr2.fit(scale(X_train),y_train_2)
svr_err = RMSLE(svr1.predict(scaler.transform(X_val)),svr2.predict(scaler.transform(X_val)),y_val_1,y_val_2)
print "SVR error: {}".format(svr_err)
svr3 = SVR(C=10,kernel='poly')
svr4 = SVR(C=10,kernel='poly')
svr3.fit(scale(X_train),y_train_1)
svr4.fit(scale(X_train),y_train_2)
svr_err = RMSLE(svr3.predict(scaler.transform(X_val)),svr4.predict(scaler.transform(X_val)),y_val_1,y_val_2)
print "poly SVR error: {}".format(svr_err)
rig1 = Ridge(alpha=80)
rig2 = Ridge(alpha=80)
rig1.fit(X_train,y_train_1)
rig2.fit(X_train,y_train_2)
rig_err = RMSLE(rig1.predict(X_val),rig2.predict(X_val),y_val_1,y_val_2)
print "Ridge error: {}".format(rig_err)
lass1 = Lasso(alpha=.01)
lass2 = Lasso(alpha=.01)
lass1.fit(X_train,y_train_1)
lass2.fit(X_train,y_train_2)
lass_err = RMSLE(lass1.predict(X_val),lass2.predict(X_val),y_val_1,y_val_2)
print "Lasso error: {}".format(lass_err)
neigh1 = KNeighborsRegressor(n_neighbors=4)
neigh2 = KNeighborsRegressor(n_neighbors=4)
neigh1.fit(X_train,y_train_1)
neigh2.fit(X_train,y_train_2)
neigh_err = RMSLE(neigh1.predict(X_val),neigh2.predict(X_val),y_val_1,y_val_2)
print "KNeighborsRegressor error: {}".format(neigh_err)
tree1 = DecisionTreeRegressor(criterion="mae", max_depth=6,min_samples_split=5)
tree2 = DecisionTreeRegressor(criterion="mae",max_depth=6,min_samples_split=5)
tree1.fit(X_train,y_train_1)
tree2.fit(X_train,y_train_2)
tree_err = RMSLE(tree1.predict(X_val),tree2.predict(X_val),y_val_1,y_val_2)
print "DecisionTreeRegressor error: {}".format(tree_err)


sample = pd.read_csv("./sample_submission.csv")
#sample["formation_energy_ev_natom"] = test_pred_1[0]
#sample["bandgap_energy_ev"] = test_pred_2[0]
#sample.to_csv("sub.csv",index=False)
print "runtime: {}".format(time.time()-start)
