import streamlit as st
from gsheetsdb import connect
import random
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title(" Dementia staging with machine learning    ")

# df = pd.read_csv("GCDR_SB_Labels")


# df = df.drop("Unnamed: 0", axis=1)


gsheet_url = "https://docs.google.com/spreadsheets/d/1c7TuS0sTm14AAJLoCAjCUP0PIw5Pye8jm5EBIqjxeT4/edit?usp=sharing"
conn = connect()
rows = conn.execute(f'SELECT * FROM "{gsheet_url}"')
df = pd.DataFrame(rows)

X = df.drop(["GCDR", "CDR_SB", "GCDR_L", "CDR_SB_L"], axis=1)
y = df["GCDR_L"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

# pred = rfc.predict(X_test)


CDMEMORY = st.slider("CDMEMORY", min_value=0.0, max_value=3.0, step=0.05)
CDORIENT = st.slider("CDORIENT", min_value=0.0, max_value=3.0, step=0.05)
CDJUDGE = st.slider("CDJUDGE", min_value=0.0, max_value=3.0, step=0.05)
CDCOMMUN = st.slider("CDCOMMUN", min_value=0.0, max_value=3.0, step=0.05)
CDHOME = st.slider("CDHOME", min_value=0.0, max_value=3.0, step=0.05)
CDCARE = st.slider("CDCARE", min_value=0.0, max_value=3.0, step=0.05)

cdr_rfc = rfc.predict([[CDMEMORY, CDORIENT, CDJUDGE, CDCOMMUN, CDHOME, CDCARE]])

st.write(cdr_rfc)
