import streamlit as st
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn                        import metrics, svm
from sklearn.linear_model           import LinearRegression



# headings
st.title("CDR:  Dementia staging with machine learning   ")

#selecionando um classificador
#algorithm = st.sidebar.selectbox("Select a classifier ",
 #                                ("Linear regression", "Logistic regression","SVM"))
#uploading data
df = pd.read_csv("CDR_prepocessed", sep=",")

def lab(x):
    if x==0.0: x="Health"
    elif x==0.5: x="Questionable"
    elif x==1: x="Mild"
    elif x==2: x="Moderate"
    else: x="Severe"

    return x

df= df.drop("Unnamed: 0",  axis=1)

df["GLOBAL_L"]=df["GLOBAL"].apply(lambda x: lab(x))
df= df.drop_duplicates( keep = "last")


X= df.drop( ["GLOBAL", "GLOBAL_L"], axis=1)

y = df["GLOBAL"]

#st.write("Shape of dataset", df.shape)
#st.write("The dataset was provided by ADNI and it is composed of ...")


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=43)


# classificadores

model_LR = LinearRegression()

model_svm = svm.SVR(kernel='linear')

# Treino : fitando os classificadores
model_LR.fit(X_train,y_train)

#model_LogR.fit(X_train,y_train)

model_svm.fit(X_train,y_train)




#def classifiers(MEMORY, ORIENT, JUDGE, COMMUN, HOME, CARE):
    #cdr_LR = model_LR.predict([[MEMORY, ORIENT, JUDGE, COMMUN, HOME, CARE]])
     #cdr_LogR = model_LogR.predict([[MEMORY, ORIENT, JUDGE, COMMUN, HOME, CARE]])
     #cdr_svm = model_svm.predict([[MEMORY, ORIENT, JUDGE, COMMUN, HOME, CARE]])
    #st.write("Os notas indicadas para domínios são ", MEMORY, ORIENT, JUDGE, COMMUN, HOME, CARE)
    #return cdr_LR, cdr_LogR, cdr_svm

CDMEMORY = st.slider("CDMEMORY", min_value=0.0, max_value=3.0, step=0.05)
CDORIENT = st.slider("CDORIENT", min_value=0.0, max_value=3.0, step=0.05)
CDJUDGE = st.slider("CDJUDGE", min_value=0.0, max_value=3.0, step=0.05)
CDCOMMUN = st.slider("CDCOMMUN", min_value=0.0, max_value=3.0, step=0.05)
CDHOME = st.slider("CDHOME", min_value=0.0, max_value=3.0, step=0.05)
CDCARE = st.slider("CDCARE", min_value=0.0, max_value=3.0, step=0.05)

#classifiers(CDMEMORY, CDORIENT,CDJUDGE,CDCOMMUN,CDHOME,CDCARE)

# estagiando

# Global score via linear regression
cdr_LR = model_LR.predict([[CDMEMORY, CDORIENT, CDJUDGE, CDCOMMUN, CDHOME, CDCARE]])

# Global score via SVM
cdr_svm = model_svm.predict([[CDMEMORY, CDORIENT, CDJUDGE, CDCOMMUN, CDHOME, CDCARE]])


st.write(""" # Estagiamento """)
st.write("GLOBAL pela Regressão Linear", cdr_LR[0])
#st.write("GLOBAL pela Regressão Logística", cdr_LogR[0])
st.write("GLOBAL pela SVM", cdr_svm[0])


df["GLOBAL_LR"] =  model_LR.predict(df.drop(["GLOBAL","GLOBAL_L" ], axis = 1))
df["GLOBAL_SVM"] =  model_svm.predict(df.drop(["GLOBAL","GLOBAL_L","GLOBAL_LR" ], axis = 1))


# Now we aree going to use KNN to stage the dementia: this occurs in two
#parts first using Global LR then after Global SVM.


####################################################################
####################################################################
#With Global LR
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

scaler = StandardScaler()

scaler.fit(df.drop(['GLOBAL','GLOBAL_L','GLOBAL_SVM'],axis=1))

scaled_features = scaler.transform(df.drop(['GLOBAL','GLOBAL_L','GLOBAL_SVM'],axis=1))


df_LR = df.drop(['GLOBAL',"GLOBAL_L","GLOBAL_SVM"], axis=1)


X_feat_LR = pd.DataFrame(scaled_features,columns=df_LR.columns)

XX_train, XX_test, yy_train, yy_test = train_test_split(X_feat_LR, df['GLOBAL_L'],

                                                        test_size=0.2, random_state=43)  # 47 mto estranho de bom

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(XX_train,yy_train)

pred = knn.predict([[CDMEMORY, CDORIENT, CDJUDGE, CDCOMMUN, CDHOME, CDCARE, cdr_LR]])

st.write("Linear regression")
st.write(pred)


# end staging with KNN and Global LR
####################################################################
####################################################################


####################################################################
####################################################################
#With Global SVM

scaler.fit(df.drop(['GLOBAL','GLOBAL_L','GLOBAL_LR'],axis=1))

scaled_features_SVM = scaler.transform(df.drop(['GLOBAL','GLOBAL_L','GLOBAL_LR'],axis=1))

df_SVM = df.drop(['GLOBAL',"GLOBAL_LR", "GLOBAL_L"], axis=1) # manobra para pegar o nome das colunas

X_feat_SVM = pd.DataFrame(scaled_features_SVM,columns=df_SVM.columns)

XXX_train, XXX_test, yyy_train, yyy_test = train_test_split(X_feat_SVM, df['GLOBAL_L'],

                                                            test_size=0.2, random_state=43)

#knn.fit(XXX_train,yyy_train)

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(XXX_train,yyy_train)
pred_SVM = knn.predict(XXX_test)


pred_SVM = knn.predict([[CDMEMORY, CDORIENT, CDJUDGE, CDCOMMUN, CDHOME, CDCARE, cdr_svm]])

st.write("SVM")
st.write(pred_SVM)

# end staging with KNN and Global SVM
####################################################################
####################################################################

# we can determine the optimum value of k when we get the highest test score for that value.
# For that, we can evaluate the training and testing scores for up to 20 nearest neighbors: