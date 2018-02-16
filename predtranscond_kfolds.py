#had problems with overfitting, so tried to implement KFold cross-analysis to reduce variance, but ran into some bugs
#that I could not get rid of 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import scale, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold

import warnings
import time
warnings.filterwarnings("ignore")
start = time.time()

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
test_noxyz = test.drop("id",axis=1)
train_noxyz = train.drop("id",axis=1)

X_train_noxyz = train.head(2400).drop("id",axis=1)
X_val_noxyz = train.tail(2400).drop("id",axis=1)
X_train_noxyz = X_train_noxyz.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)
X_val_noxyz = X_val_noxyz.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)
test_noxyz_train = test_noxyz
test_noxyz_val = test_noxyz

def get_xyz_data(filename):
    pos_data=[]
    lat_data=[]
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)

def RMSLE(pred1,pred2,true1,true2):
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    true1 = np.array(true1)
    true2 = np.array(true2)
    error1 = np.square(np.log(pred1 + 1) - np.log(true1 + 1)).mean()**0.5
    error2 = np.square(np.log(pred2 + 1) - np.log(true2 + 1)).mean()**0.5
    return np.sum(error1 + error2)/2

kf = KFold(n_splits=5) 
def get_oof(clf1, clf2, x_train, x_test, y_train1, y_train2):
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train1 = np.asarray(y_train1)
    y_train2 = np.asarray(y_train2)
    oof_train1 = np.zeros(x_train.shape[0])
    oof_test1 = np.zeros(x_test.shape[0])
    oof_test_skf1 = np.empty([5, 600])
    oof_train2 = np.zeros(x_train.shape[0])
    oof_test2 = np.zeros(x_test.shape[0])
    oof_test_skf2 = np.empty([5, 600])

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr1 = y_train1[train_index]
        y_tr2 = y_train2[train_index]
        x_te = x_train[test_index]

        print type(np.asarray(x_tr))
        print type(np.asarray(y_tr1))
        clf1.fit(np.asarray(x_tr), np.asarray(y_tr1))
        clf2.fit(np.asarray(x_tr), np.asarray(y_tr2))

        oof_train1[test_index] = clf1.predict(x_te)
        oof_train2[test_index] = clf2.predict(x_te)
        oof_test_skf1[i,:] = clf1.predict(x_test)
        oof_test_skf2[i,:] = clf2.predict(x_test)
        svr_err = RMSLE(oof_train1[test_index],oof_train2[test_index],y_train1[test_index],y_train2[test_index])
        print(svr_err)

    oof_test1[:] = oof_test_skf1.mean(axis=0)
    oof_test2[:] = oof_test_skf2.mean(axis=0)
    return pd.Series(oof_train1.reshape(-1,1).tolist()), pd.Series(oof_train2.reshape(-1,1).tolist()), pd.Series(oof_test1.reshape(-1,1).tolist()), pd.Series(oof_test2.reshape(-1,1).tolist())

ga_cols = []
al_cols = []
o_cols =[]
in_cols = []

for i in range(6):
    ga_cols.append("Ga_"+str(i))
    al_cols.append("Al_"+str(i))
    o_cols.append("O_"+str(i))
    in_cols.append("In_"+str(i))

ga_df = pd.DataFrame(columns=ga_cols)
al_df = pd.DataFrame(columns=al_cols)
o_df = pd.DataFrame(columns=o_cols)
in_df = pd.DataFrame(columns=in_cols)

for i in train.id.values:
    fn = "./train/{}/geometry.xyz".format(i)
    train_xyz, train_lat = get_xyz_data(fn)

    ga_list = []
    al_list = []
    o_list=[]
    in_list=[]

    for li in train_xyz:
        if li[1] == "Ga":
            ga_list.append(li[0])
        if li[1] == "Al":
            al_list.append(li[0])
        if li[1] == "In":
            in_list.append(li[0])
        if li[1] == "O":
            o_list.append(li[0])


    #dimensionality reduction using PCA
    try:
        model = PCA(n_components=2)
        ga_list = np.array(ga_list)
        temp_ga = model.fit_transform(ga_list.transpose())
        temp_ga = [item for sublist in temp_ga for item in sublist]
    except:
        temp_ga = [0,0,0,0,0,0]
    try:
        model = PCA(n_components=2)
        al_list = np.array(al_list)
        temp_al = model.fit_transform(al_list.transpose())
        temp_al = [item for sublist in temp_al for item in sublist]
    except:
        temp_al = [0,0,0,0,0,0]
    try:
        model = PCA(n_components=2)
        o_list = np.array(o_list)
        temp_o = model.fit_transform(o_list.transpose())
        temp_o = [item for sublist in temp_o for item in sublist]
    except:
        temp_o = [0,0,0,0,0,0]
    try:
        model = PCA(n_components=2)
        in_list = np.array(in_list)
        temp_in = model.fit_transform(in_list.transpose())
        temp_in = [item for sublist in temp_in for item in sublist]
    except:
        temp_in = [0,0,0,0,0,0]

    temp_ga = pd.DataFrame(temp_ga).transpose()
    temp_ga.columns = ga_cols
    temp_ga.index = np.array([i])

    temp_al = pd.DataFrame(temp_al).transpose()
    temp_al.columns = al_cols
    temp_al.index = np.array([i])

    temp_o = pd.DataFrame(temp_o).transpose()
    temp_o.columns = o_cols
    temp_o.index = np.array([i])
    temp_in = pd.DataFrame(temp_in).transpose()
    temp_in.columns = in_cols
    temp_in.index = np.array([i])

    ga_df = pd.concat([ga_df,temp_ga])
    al_df = pd.concat([al_df,temp_al])
    o_df = pd.concat([o_df,temp_o])
    in_df = pd.concat([in_df,temp_in])

ga_df["id"] = ga_df.index
al_df["id"] = al_df.index
o_df["id"] = o_df.index
in_df["id"] = in_df.index

train = pd.merge(train,ga_df,on=["id"],how="left")
train = pd.merge(train,al_df,on=["id"],how="left")
train = pd.merge(train,o_df,on=["id"],how="left")
train = pd.merge(train,in_df,on=["id"],how="left")

ga_df = pd.DataFrame(columns=ga_cols)
al_df = pd.DataFrame(columns=al_cols)
o_df = pd.DataFrame(columns=o_cols)
in_df = pd.DataFrame(columns=in_cols)

for i in test.id.values:
    fn = "./test/{}/geometry.xyz".format(i)
    test_xyz, test_lat = get_xyz_data(fn)

    ga_list = []
    al_list = []
    o_list=[]
    in_list=[]

    for li in test_xyz:
        if li[1] == "Ga":
            ga_list.append(li[0])
        if li[1] == "Al":
            al_list.append(li[0])
        if li[1] == "In":
            in_list.append(li[0])
        if li[1] == "O":
            o_list.append(li[0])


    #dimensionality reduction using PCA
    try:
        model = PCA(n_components=2)
        ga_list = np.array(ga_list)
        temp_ga = model.fit_transform(ga_list.transpose())
        temp_ga = [item for sublist in temp_ga for item in sublist]
    except:
        temp_ga = [0,0,0,0,0,0]
    try:
        model = PCA(n_components=2)
        al_list = np.array(al_list)
        temp_al = model.fit_transform(al_list.transpose())
        temp_al = [item for sublist in temp_al for item in sublist]
    except:
        temp_al = [0,0,0,0,0,0]
    try:
        model = PCA(n_components=2)
        o_list = np.array(o_list)
        temp_o = model.fit_transform(o_list.transpose())
        temp_o = [item for sublist in temp_o for item in sublist]
    except:
        temp_o = [0,0,0,0,0,0]
    try:
        model = PCA(n_components=2)
        in_list = np.array(in_list)
        temp_in = model.fit_transform(in_list.transpose())
        temp_in = [item for sublist in temp_in for item in sublist]
    except:
        temp_in = [0,0,0,0,0,0]

    temp_ga = pd.DataFrame(temp_ga).transpose()
    temp_ga.columns = ga_cols
    temp_ga.index = np.array([i])

    temp_al = pd.DataFrame(temp_al).transpose()
    temp_al.columns = al_cols
    temp_al.index = np.array([i])

    temp_o = pd.DataFrame(temp_o).transpose()
    temp_o.columns = o_cols
    temp_o.index = np.array([i])
    temp_in = pd.DataFrame(temp_in).transpose()
    temp_in.columns = in_cols
    temp_in.index = np.array([i])

    ga_df = pd.concat([ga_df,temp_ga])
    al_df = pd.concat([al_df,temp_al])
    o_df = pd.concat([o_df,temp_o])
    in_df = pd.concat([in_df,temp_in])

ga_df["id"] = ga_df.index
al_df["id"] = al_df.index
o_df["id"] = o_df.index
in_df["id"] = in_df.index

test = pd.merge(test,ga_df,on=["id"],how="left")
test = pd.merge(test,al_df,on=["id"],how="left")
test = pd.merge(test,o_df,on=["id"],how="left")
test = pd.merge(test,in_df,on=["id"],how="left")

#X_train = train.head(1200).drop("id",axis=1)
#X_val = train.tail(1200).drop("id",axis=1)
X_train = train.drop("id",axis=1)
X_val = train.drop("id",axis=1)

X_train = train.drop("id",axis=1)
X_val = train.drop("id",axis=1)
y_train_1 = X_train["formation_energy_ev_natom"]
y_train_2 = X_train["bandgap_energy_ev"]
y_val_1 = X_val["formation_energy_ev_natom"]
y_val_2 = X_val["bandgap_energy_ev"]
X_train = X_train.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)
X_val = X_val.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)

X_train.In_0 = X_train.In_0.astype("float")
X_train.In_1 = X_train.In_1.astype("float")
X_train.In_2 = X_train.In_2.astype("float")
X_train.In_3 = X_train.In_3.astype("float")
X_train.In_4 = X_train.In_4.astype("float")
X_train.In_5 = X_train.In_5.astype("float")

X_val.In_0 = X_val.In_0.astype("float")
X_val.In_1 = X_val.In_1.astype("float")
X_val.In_2 = X_val.In_2.astype("float")
X_val.In_3 = X_val.In_3.astype("float")
X_val.In_4 = X_val.In_4.astype("float")
X_val.In_5 = X_val.In_5.astype("float")

test.In_0 = test.In_0.astype("float")
test.In_1 = test.In_1.astype("float")
test.In_2 = test.In_2.astype("float")
test.In_3 = test.In_3.astype("float")
test.In_4 = test.In_4.astype("float")
test.In_5 = test.In_5.astype("float")

scaler = StandardScaler().fit(X_train_noxyz)
train2f = pd.DataFrame()
train2bg = pd.DataFrame()
test2f = pd.DataFrame()
test2bg = pd.DataFrame()
base_err = RMSLE(np.random.rand(1,len(y_train_1))*0.23,np.random.rand(1,len(y_train_2))*2.5,y_train_1,y_train_2)
print "baseline error: {}".format(base_err)

test_val = test.drop("id",axis=1)

svr1 = SVR(C=10)
svr2 = SVR(C=10)
train2f["svr"], train2bg["svr"], test2f["svr"], test2bg["svr"] = get_oof(svr1, svr2, 
        scale(X_train_noxyz), scale(test_noxyz_val), y_train_1, y_train_2)
#svr1.fit(scale(X_train_noxyz),y_train_1)
#svr2.fit(scale(X_train_noxyz),y_train_2)
#svr_err = RMSLE(svr1.predict(scaler.transform(X_val_noxyz)),svr2.predict(scaler.transform(X_val_noxyz)),y_val_1,y_val_2)
#print "SVR error: {}".format(svr_err)
#train2f["svr"] = svr1.predict(scaler.transform(X_val_noxyz)) 
#train2bg["svr"] = svr2.predict(scaler.transform(X_val_noxyz))
#test2bg["svr"] = svr2.predict(scaler.transform(test_noxyz_val))
#test2f["svr"] = svr1.predict(scaler.transform(test_noxyz_val))

rig1 = Ridge(alpha=80)
rig2 = Ridge(alpha=80)
train2f["rig"], train2bg["rig"], test2f["rig"], test2bg["rig"] = get_oof(rig1, rig2, 
        X_train, test_val, y_train_1, y_train_2)
#rig1.fit(X_train,y_train_1)
#rig2.fit(X_train,y_train_2)
#rig_err = RMSLE(rig1.predict(X_val),rig2.predict(X_val),y_val_1,y_val_2)
#print "Ridge error: {}".format(rig_err)
#train2f["ridge"] = rig1.predict(X_val)
#train2bg["ridge"] = rig2.predict(X_val)
#test2bg["ridge"] = rig1.predict(test_val)
#test2f["ridge"] = rig2.predict(test_val)

lass1 = Lasso(alpha=.01)
lass2 = Lasso(alpha=.01)
train2f["lass"], train2bg["lass"], test2f["lass"], test2bg["lass"] = get_oof(lass1, lass2, 
        X_train, test_val, y_train_1, y_train_2)
#lass1.fit(X_train,y_train_1)
#lass2.fit(X_train,y_train_2)
#lass_err = RMSLE(lass1.predict(X_val),lass2.predict(X_val),y_val_1,y_val_2)
#print "Lasso error: {}".format(lass_err)
#train2f["lasso"] = lass1.predict(X_val)
#train2bg["lasso"] = lass2.predict(X_val)
#test2bg["lasso"] = lass1.predict(test_val)
#test2f["lasso"] = lass2.predict(test_val)

neigh1 = KNeighborsRegressor(n_neighbors=4)
neigh2 = KNeighborsRegressor(n_neighbors=4)
train2f["neigh"], train2bg["neigh"], test2f["neigh"], test2bg["neigh"] = get_oof(neigh1, neigh2, 
        X_train, test_val, y_train_1, y_train_2)
#neigh1.fit(X_train,y_train_1)
#neigh2.fit(X_train,y_train_2)
#neigh_err = RMSLE(neigh1.predict(X_val),neigh2.predict(X_val),y_val_1,y_val_2)
#print "KNeighborsRegressor error: {}".format(neigh_err)
#train2f["neigh"] = neigh1.predict(X_val)
#train2bg["neigh"] = neigh2.predict(X_val)
#test2bg["neigh"] = neigh1.predict(test_val)
#test2f["neigh"] = neigh2.predict(test_val)

tree1 = DecisionTreeRegressor(criterion="mae", max_depth=6,min_samples_split=5,)
tree2 = DecisionTreeRegressor(criterion="mae",max_depth=6,min_samples_split=5,)
train2f["tree"], train2bg["tree"], test2f["tree"], test2bg["tree"] = get_oof(tree1, tree2, 
        X_train, test_val, y_train_1, y_train_2)
#tree1.fit(X_train,y_train_1)
#tree2.fit(X_train,y_train_2)
#tree_err = RMSLE(tree1.predict(X_val),tree2.predict(X_val),y_val_1,y_val_2)
#print "DecisionTreeRegressor error: {}".format(tree_err)
#train2f["dtr"] = tree1.predict(X_val)
#train2bg["dtr"] = tree2.predict(X_val)
#test2bg["dtr"] = tree1.predict(test_val)
#test2f["dtr"] = tree2.predict(test_val)

#for h, i in enumerate([None, DecisionTreeRegressor(), KNeighborsRegressor()]): 
#    bag1 = BaggingRegressor(base_estimator=i, max_samples=.5,max_features=.5,bootstrap=False)
#    bag2 = BaggingRegressor(base_estimator=i, max_samples=.5,max_features=.5,bootstrap=False)
#    train2f["bag{}".format(h)], train2bg["bag{}".format(h)], test2f["bag{}".format(h)], test2bg["bag{}".format(h)] = get_oof(bag1, bag2, X_train, test_val, y_train_1, y_train_2)
#    #bag1.fit(X_train,y_train_1)
#    #bag2.fit(X_train,y_train_2)
#    #bag_err = RMSLE(bag1.predict(X_val,bag2.predict(X_val,y_val_1,y_val_2)
#    #print "Bagging error: {}".format(bag_err)
#    #train2f["bag{}".format(h)] = bag1.predict(X_val)
#    #train2bg["bag{}".format(h)] = bag2.predict(X_val)
#    #test2bg["bag{}".format(h)] = bag1.predict(test_val)
#    #test2f["bag{}".format(h)] = bag2.predict(test_val)
#
#    ada1 = AdaBoostRegressor(base_estimator=i,n_estimators=100)
#    ada2 = AdaBoostRegressor(base_estimator=i,n_estimators=100)
#    train2f["ada{}".format(h)], train2bg["ada{}".format(h)], test2f["ada{}".format(h)], test2bg["ada{}".format(h)] = get_oof(ada1, ada2, X_train, test_val, y_train_1, y_train_2)
#    #ada1.fit(X_train,y_train_1)
#    #ada2.fit(X_train,y_train_2)
#    #ada_err = RMSLE(ada1.predict(X_val,bag2.predict(X_val,y_val_1,y_val_2)
#    #print "Ada error: {}".format(ada_err)
#    #train2f["ada{}".format(h)] = ada1.predict(X_val)
#    #train2bg["ada{}".format(h)] = ada2.predict(X_val)
#    #test2bg["ada{}".format(h)] = ada1.predict(test_val)
#    #test2f["ada{}".format(h)] = ada2.predict(test_val)

forest1 = RandomForestRegressor(n_estimators=20,max_features=1,bootstrap=False)
forest2 = RandomForestRegressor(n_estimators=20,max_features=1,bootstrap=False)
train2f["forest"], train2bg["forest"], test2f["forest"], test2bg["forest"] = get_oof(forest1, forest2, 
        X_train, test_val, y_train_1, y_train_2)
#forest1.fit(X_train,y_train_1)
#forest2.fit(X_train,y_train_2)
#forest_err = RMSLE(forest1.predict(X_val,bag2.predict(X_val,y_val_1,y_val_2)
#print "RandomForest error: {}".format(forest_err)
#train2f["forest"] = forest1.predict(X_val)
#train2bg["forest"] = forest2.predict(X_val)
#test2bg["forest"] = forest1.predict(test_val)
#test2f["forest"] = forest2.predict(test_val)

X_train2f = train2f.head(2400)
X_train2bg = train2bg.head(2400)
X_val2f = train2f.tail(200)
X_val2bg = train2bg.tail(200)
#y_train2f = y_val_1.tail(2400).head(2400)
#y_train2bg = y_val_2.tail(2400).head(2400)
y_train2f = y_val_1.head(2400)
y_train2bg = y_val_2.head(2400)
y_val2f = y_val_1.tail(200)
y_val2bg = y_val_2.tail(200)

X_train3f = pd.DataFrame()
X_train3bg = pd.DataFrame()
test3f = pd.DataFrame()
test3bg = pd.DataFrame()

#gbm1 = xgb.XGBRegressor(n_estimators=1000,max_depth=3,min_child_weight=2,
#        gamma=.1,subsample=.9,colsample_bytree=.8,objective='reg:linear',scale_pos_weight=1)
#gbm2 = xgb.XGBRegressor(n_estimators=1000,max_depth=3,min_child_weight=2,
#        gamma=.1,subsample=.9,colsample_bytree=.8,objective='reg:linear',scale_pos_weight=1)
#X_train3f["gbm"], X_train3bg["gbm"], test3f["gbm"], test3bg["gbm"] = get_oof(gbm1, gbm2, 
#        train2f, train2bg, y_train_1, y_train_2)
#gbm1.fit(X_train2f,y_train2f)
#gbm2.fit(X_train2bg,y_train2bg)
#gbm_err = RMSLE(gbm1.predict(X_val2f),gbm2.predict(X_val2bg,y_val2f,y_val2bg)
#print "XGB error: {}".format(gbm_err)
#X_train3f["xgb"] = gbm1.predict(X_val2f)
#X_train3bg["xgb"] = gbm2.predict(X_val2bg)
#test3bg["xgb"] = gbm1.predict(test2bg)
#test3f["xgb"] = gbm2.predict(test2f)

print np.asarray(train2f).shape
print np.asarray(train2bg).shape
print np.asarray(y_train2f).shape
print np.asarray(y_train2bg).shape

for h, i in enumerate([None, DecisionTreeRegressor(), KNeighborsRegressor()]): 
    bag1 = BaggingRegressor(base_estimator=i, max_samples=.5,max_features=.5,bootstrap=False)
    bag2 = BaggingRegressor(base_estimator=i, max_samples=.5,max_features=.5,bootstrap=False)
#    X_train3f["bag".format(h)], X_train3bg["bag".format(h)], test3f["bag".format(h)], test3bg["bag".format(h)] = get_oof(bag1, bag2, train2f, train2bg, y_train2f, y_train2bg)
    bag1.fit(X_train2f,y_train2f)
    bag2.fit(X_train2bg,y_train2bg)
    #bag_err = RMSLE(bag1.predict(X_val2f),bag2.predict(X_val2bg,y_val2f,y_val2bg)
    #print "Bagging error: {}".format(bag_err)
    X_train3f["bag{}".format(h)] = bag1.predict(X_val2f)
    X_train3bg["bag{}".format(h)] = bag2.predict(X_val2bg)
    test3bg["bag{}".format(h)] = bag1.predict(test2bg)
    test3f["bag{}".format(h)] = bag2.predict(test2f)

    ada1 = AdaBoostRegressor(base_estimator=i,n_estimators=100)
    ada2 = AdaBoostRegressor(base_estimator=i,n_estimators=100)
    #X_train3f["ada".format(h)], X_train3bg["ada".format(h)], test3f["ada".format(h)], test3bg["ada".format(h)] = get_oof(ada1, ada2, train2f, train2bg, y_train2f, y_train2bg)
    ada1.fit(X_train2f,y_train2f)
    ada2.fit(X_train2bg,y_train2bg)
    #ada_err = RMSLE(ada1.predict(X_val2f),bag2.predict(X_val2bg,y_val2f,y_val2bg)
    #print "Ada error: {}".format(ada_err)
    X_train3f["ada{}".format(h)] = ada1.predict(X_val2f)
    X_train3bg["ada{}".format(h)] = ada2.predict(X_val2bg)
    test3bg["ada{}".format(h)] = ada1.predict(test2bg)
    test3f["ada{}".format(h)] = ada2.predict(test2f)

forest1 = RandomForestRegressor(n_estimators=20,max_features=1,bootstrap=False)
forest2 = RandomForestRegressor(n_estimators=20,max_features=1,bootstrap=False)
#X_train3f["forest"], X_train3bg["forest"], test3f["forest"], test3bg["forest"] = get_oof(forest1, forest2, 
#        train2f, train2bg, y_train2f, y_train2bg)
forest1.fit(X_train2f,y_train2f)
forest2.fit(X_train2bg,y_train2bg)
#forest_err = RMSLE(forest1.predict(X_val2f),bag2.predict(X_val2bg,y_val2f,y_val2bg)
#print "RandomForest error: {}".format(forest_err)
X_train3f["forest"] = forest1.predict(X_val2f)
X_train3bg["forest"] = forest2.predict(X_val2bg)
test3bg["forest"] = forest1.predict(test2bg)
test3f["forest"] = forest2.predict(test2f)

X_train3f = X_train3f.mean(axis=1) 
X_train3bg = X_train3bg.mean(axis=1)
test3f = test3f.mean(axis=1)
test3bg = test3bg.mean(axis=1)
#final_err = RMSLE(X_train3f,X_train3bg,y_val2f,y_val2bg)
#print "averaged results error: {}".format(final_err)

sample = pd.read_csv("./sample_submission.csv")
sample["formation_energy_ev_natom"] = test3f
sample["bandgap_energy_ev"] = test3bg
sample.to_csv("sub.csv",index=False)
print "runtime: {}".format(time.time()-start)
