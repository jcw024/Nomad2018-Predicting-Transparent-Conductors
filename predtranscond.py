import numpy as np
import pandas as pd
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

X_train_noxyz = train.head(2000).drop("id",axis=1)
X_val_noxyz = train.tail(400).drop("id",axis=1)
X_train_noxyz = X_train_noxyz.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)
X_val_noxyz = X_val_noxyz.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)

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

X_train = train.head(2000).drop("id",axis=1)
X_val = train.tail(400).drop("id",axis=1)

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
test = test.drop("id",axis=1)
base_err = RMSLE(np.random.rand(1,len(y_train_1))*0.23,np.random.rand(1,len(y_train_2))*2.5,y_train_1,y_train_2)
print "baseline error: {}".format(base_err)

svr1 = SVR(C=10)
svr2 = SVR(C=10)
svr1.fit(scale(X_train_noxyz),y_train_1)
svr2.fit(scale(X_train_noxyz),y_train_2)
svr_err = RMSLE(svr1.predict(scaler.transform(X_val_noxyz)),svr2.predict(scaler.transform(X_val_noxyz)),y_val_1,y_val_2)
print "SVR error: {}".format(svr_err)
train2f["svr"] = svr1.predict(scaler.transform(X_val_noxyz)) 
train2bg["svr"] = svr2.predict(scaler.transform(X_val_noxyz))
test2f["svr"] = svr1.predict(scaler.transform(test_noxyz))
test2bg["svr"] = svr2.predict(scaler.transform(test_noxyz))

#poor accuracy using poly, so removed model
#svr3 = SVR(C=10,kernel='poly')
#svr4 = SVR(C=10,kernel='poly')
#svr3.fit(scale(X_train_noxyz),y_train_1)
#svr4.fit(scale(X_train_noxyz),y_train_2)
#svr_err = RMSLE(svr3.predict(scaler.transform(X_val_noxyz)),svr4.predict(scaler.transform(X_val_noxyz)),y_val_1,y_val_2)
#print "poly SVR error: {}".format(svr_err)
#train2f["svr_p"] = svr3.predict(scaler.transform(X_val_noxyz))
#train2bg["svr_p"] = svr4.predict(scaler.transform(X_val_noxyz))

rig1 = Ridge(alpha=80)
rig2 = Ridge(alpha=80)
rig1.fit(X_train,y_train_1)
rig2.fit(X_train,y_train_2)
rig_err = RMSLE(rig1.predict(X_val),rig2.predict(X_val),y_val_1,y_val_2)
print "Ridge error: {}".format(rig_err)
train2f["ridge"] = rig1.predict(X_val)
train2bg["ridge"] = rig2.predict(X_val)
test2f["ridge"] = rig1.predict(test)
test2bg["ridge"] = rig2.predict(test)

lass1 = Lasso(alpha=.01)
lass2 = Lasso(alpha=.01)
lass1.fit(X_train,y_train_1)
lass2.fit(X_train,y_train_2)
lass_err = RMSLE(lass1.predict(X_val),lass2.predict(X_val),y_val_1,y_val_2)
print "Lasso error: {}".format(lass_err)
train2f["lasso"] = lass1.predict(X_val)
train2bg["lasso"] = lass2.predict(X_val)
test2f["lasso"] = rig1.predict(test)
test2bg["lasso"] = rig2.predict(test)


neigh1 = KNeighborsRegressor(n_neighbors=4)
neigh2 = KNeighborsRegressor(n_neighbors=4)
neigh1.fit(X_train_noxyz,y_train_1)
neigh2.fit(X_train_noxyz,y_train_2)
neigh_err = RMSLE(neigh1.predict(X_val_noxyz),neigh2.predict(X_val_noxyz),y_val_1,y_val_2)
print "KNeighborsRegressor error: {}".format(neigh_err)
train2f["neigh"] = neigh1.predict(X_val_noxyz)
train2bg["neigh"] = neigh2.predict(X_val_noxyz)
test2f["neigh"] = neigh1.predict(test_noxyz)
test2bg["neigh"] = neigh2.predict(test_noxyz)


tree1 = DecisionTreeRegressor(criterion="mae", max_depth=6,min_samples_split=5,)
tree2 = DecisionTreeRegressor(criterion="mae",max_depth=6,min_samples_split=5,)
tree1.fit(X_train,y_train_1)
tree2.fit(X_train,y_train_2)
tree_err = RMSLE(tree1.predict(X_val),tree2.predict(X_val),y_val_1,y_val_2)
print "DecisionTreeRegressor error: {}".format(tree_err)
train2f["dtr"] = tree1.predict(X_val)
train2bg["dtr"] = tree2.predict(X_val)
test2f["dtr"] = tree1.predict(test)
test2bg["dtr"] = tree2.predict(test)


for h, i in enumerate([None, DecisionTreeRegressor(), KNeighborsRegressor()]): 
    bag1 = BaggingRegressor(base_estimator=i, max_samples=.5,max_features=.5,bootstrap=False)
    bag2 = BaggingRegressor(base_estimator=i, max_samples=.5,max_features=.5,bootstrap=False)
    bag1.fit(X_train,y_train_1)
    bag2.fit(X_train,y_train_2)
    bag_err = RMSLE(bag1.predict(X_val),bag2.predict(X_val),y_val_1,y_val_2)
    print "Bagging error: {}".format(bag_err)
    train2f["bag{}".format(h)] = bag1.predict(X_val)
    train2bg["bag{}".format(h)] = bag2.predict(X_val)
    test2f["bag{}".format(h)] = bag1.predict(test)
    test2bg["bag{}".format(h)] = bag2.predict(test)

    ada1 = AdaBoostRegressor(base_estimator=i,n_estimators=100)
    ada2 = AdaBoostRegressor(base_estimator=i,n_estimators=100)
    ada1.fit(X_train,y_train_1)
    ada2.fit(X_train,y_train_2)
    ada_err = RMSLE(ada1.predict(X_val),bag2.predict(X_val),y_val_1,y_val_2)
    print "Ada error: {}".format(ada_err)
    train2f["ada{}".format(h)] = ada1.predict(X_val)
    train2bg["ada{}".format(h)] = ada2.predict(X_val)
    test2f["ada{}".format(h)] = ada1.predict(test)
    test2bg["ada{}".format(h)] = ada2.predict(test)


forest1 = RandomForestRegressor(n_estimators=20,max_features=1,bootstrap=False)
forest2 = RandomForestRegressor(n_estimators=20,max_features=1,bootstrap=False)
forest1.fit(X_train,y_train_1)
forest2.fit(X_train,y_train_2)
forest_err = RMSLE(forest1.predict(X_val),bag2.predict(X_val),y_val_1,y_val_2)
print "RandomForest error: {}".format(forest_err)
train2f["forest"] = forest1.predict(X_val)
train2bg["forest"] = forest2.predict(X_val)
test2f["forest"] = forest1.predict(test)
test2bg["forest"] = forest2.predict(test)


X_train2f = train2f.head(320)
X_train2bg = train2bg.head(320)
X_val2f = train2f.tail(80)
X_val2bg = train2bg.tail(80)
y_train2f = y_val_1.tail(400).head(320)
y_train2bg = y_val_2.tail(400).head(320)
y_val2f = y_val_1.tail(80)
y_val2bg = y_val_2.tail(80)

X_train3f = pd.DataFrame()
X_train3bg = pd.DataFrame()
test3f = pd.DataFrame()
test3bg = pd.DataFrame()

gbm1 = xgb.XGBRegressor(n_estimators=1000,max_depth=4,min_child_weight=2,
        gamma=.1,subsample=.3,colsample_bytree=.3,objective='reg:linear',scale_pos_weight=1)
gbm2 = xgb.XGBRegressor(n_estimators=1000,max_depth=4,min_child_weight=2,
        gamma=.1,subsample=.3,colsample_bytree=.3,objective='reg:linear',scale_pos_weight=1)
gbm1.fit(X_train2f,y_train2f)
gbm2.fit(X_train2bg,y_train2bg)
gbm_err = RMSLE(gbm1.predict(X_val2f),gbm2.predict(X_val2bg),y_val2f,y_val2bg)
print "XGB error: {}".format(gbm_err)
X_train3f["xgb"] = gbm1.predict(X_val2f)
X_train3bg["xgb"] = gbm2.predict(X_val2bg)
test3f["xgb"] = gbm1.predict(test2f)
test3bg["xgb"] = gbm2.predict(test2bg)



for h, i in enumerate([None, DecisionTreeRegressor(), KNeighborsRegressor()]): 
    bag1 = BaggingRegressor(base_estimator=i, max_samples=.2,max_features=.2,bootstrap=False)
    bag2 = BaggingRegressor(base_estimator=i, max_samples=.2,max_features=.2,bootstrap=False)
    bag1.fit(X_train2f,y_train2f)
    bag2.fit(X_train2bg,y_train2bg)
    bag_err = RMSLE(bag1.predict(X_val2f),bag2.predict(X_val2bg),y_val2f,y_val2bg)
    print "Bagging error: {}".format(bag_err)
    X_train3f["bag{}".format(h)] = bag1.predict(X_val2f)
    X_train3bg["bag{}".format(h)] = bag2.predict(X_val2bg)
    test3f["bag"] = bag1.predict(test2f)
    test3bg["bag"] = bag2.predict(test2bg)

    ada1 = AdaBoostRegressor(base_estimator=i,n_estimators=200)
    ada2 = AdaBoostRegressor(base_estimator=i,n_estimators=200)
    ada1.fit(X_train2f,y_train2f)
    ada2.fit(X_train2bg,y_train2bg)
    ada_err = RMSLE(ada1.predict(X_val2f),bag2.predict(X_val2bg),y_val2f,y_val2bg)
    print "Ada error: {}".format(ada_err)
    X_train3f["ada{}".format(h)] = ada1.predict(X_val2f)
    X_train3bg["ada{}".format(h)] = ada2.predict(X_val2bg)
    test3f["ada"] = ada1.predict(test2f)
    test3bg["ada"] = ada2.predict(test2bg)

forest1 = RandomForestRegressor(n_estimators=200,max_features=1,bootstrap=False)
forest2 = RandomForestRegressor(n_estimators=200,max_features=1,bootstrap=False)
forest1.fit(X_train2f,y_train2f)
forest2.fit(X_train2bg,y_train2bg)
forest_err = RMSLE(forest1.predict(X_val2f),bag2.predict(X_val2bg),y_val2f,y_val2bg)
print "RandomForest error: {}".format(forest_err)
X_train3f["forest"] = forest1.predict(X_val2f)
X_train3bg["forest"] = forest2.predict(X_val2bg)
test3f["forest"] = forest1.predict(test2f)
test3bg["forest"] = forest2.predict(test2bg)

X_train3f = X_train3f.mean(axis=1) 
X_train3bg = X_train3bg.mean(axis=1)
final_err = RMSLE(X_train3f,X_train3bg,y_val2f,y_val2bg)
print "averaged results error: {}".format(final_err)

sample = pd.read_csv("./sample_submission.csv")
sample.to_csv("sub.csv",index=False)
sample["formation_energy_ev_natom"] = test3f
sample["bandgap_energy_ev"] = test3bg
sample.to_csv("sub.csv",index=False)
print "runtime: {}".format(time.time()-start)
