#file to visualize influence of more training data. generates some plots of cross-validated accuracy
#vs. number of training samples.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
import time
warnings.filterwarnings("ignore")
start = time.time()

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

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

svr_err1=[]
svr_err2=[]
rig_err=[]
lass_err=[]
neigh_err=[]
tree_err=[]

tsvr_err1=[]
tsvr_err2=[]
trig_err=[]
tlass_err=[]
tneigh_err=[]
ttree_err=[]
samplenum = np.array(range(10,23))*100

for i in samplenum:
    num = 2400 - i
    X_train = train.head(i).drop("id",axis=1)
    X_val = train.tail(num).drop("id",axis=1)
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

    svr1 = SVR(C=10)
    svr2 = SVR(C=10)
    svr1.fit(scale(X_train),y_train_1)
    svr2.fit(scale(X_train),y_train_2)
    svr_err1.append(RMSLE(svr1.predict(scale(X_val)),svr2.predict(scale(X_val)),y_val_1,y_val_2))
    tsvr_err1.append(RMSLE(svr1.predict(scale(X_train)),svr2.predict(scale(X_train)),y_train_1,y_train_2))

    svr3 = SVR(C=10,kernel='poly')
    svr4 = SVR(C=10,kernel='poly')
    svr3.fit(scale(X_train),y_train_1)
    svr4.fit(scale(X_train),y_train_2)
    svr_err2.append(RMSLE(svr3.predict(scale(X_val)),svr4.predict(scale(X_val)),y_val_1,y_val_2))
    tsvr_err2.append(RMSLE(svr3.predict(scale(X_train)),svr4.predict(scale(X_train)),y_train_1,y_train_2))

    rig1 = Ridge(alpha=80)
    rig2 = Ridge(alpha=80)
    rig1.fit(X_train,y_train_1)
    rig2.fit(X_train,y_train_2)
    rig_err.append(RMSLE(rig1.predict(X_val),rig2.predict(X_val),y_val_1,y_val_2))
    trig_err.append(RMSLE(rig1.predict(X_train),rig2.predict(X_train),y_train_1,y_train_2))
   
    lass1 = Lasso(alpha=.01)
    lass2 = Lasso(alpha=.01)
    lass1.fit(X_train,y_train_1)
    lass2.fit(X_train,y_train_2)
    lass_err.append(RMSLE(lass1.predict(X_val),lass2.predict(X_val),y_val_1,y_val_2))
    tlass_err.append(RMSLE(lass1.predict(X_train),lass2.predict(X_train),y_train_1,y_train_2))
   
    neigh1 = KNeighborsRegressor(n_neighbors=4)
    neigh2 = KNeighborsRegressor(n_neighbors=4)
    neigh1.fit(X_train,y_train_1)
    neigh2.fit(X_train,y_train_2)
    neigh_err.append(RMSLE(neigh1.predict(X_val),neigh2.predict(X_val),y_val_1,y_val_2))
    tneigh_err.append(RMSLE(neigh1.predict(X_train),neigh2.predict(X_train),y_train_1,y_train_2))

    tree1 = DecisionTreeRegressor(criterion="mae", max_depth=6,min_samples_split=5,)
    tree2 = DecisionTreeRegressor(criterion="mae",max_depth=6,min_samples_split=5,)
    tree1.fit(X_train,y_train_1)
    tree2.fit(X_train,y_train_2)
    tree_err.append(RMSLE(tree1.predict(X_val),tree2.predict(X_val),y_val_1,y_val_2))
    ttree_err.append(RMSLE(tree1.predict(X_train),tree2.predict(X_train),y_train_1,y_train_2))

plt.plot(samplenum,svr_err1,samplenum,tsvr_err1)
plt.savefig('SVR_rbf.png')
plt.clf()
plt.plot(samplenum,svr_err2,samplenum,tsvr_err2)
plt.savefig('SVR_poly.png')
plt.clf()
plt.plot(samplenum,rig_err,samplenum,trig_err)
plt.savefig('Ridge.png')
plt.clf()
plt.plot(samplenum,lass_err,samplenum,tlass_err)
plt.savefig('Lasso.png')
plt.clf()
plt.plot(samplenum,neigh_err,samplenum,tneigh_err)
plt.savefig('KNR.png')
plt.clf()
plt.plot(samplenum,tree_err,samplenum,ttree_err)
plt.savefig('DTR.png')
print "runtime: {}".format(time.time()-start)
