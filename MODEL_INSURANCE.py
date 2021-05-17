# %%
import pickle
loaded_model = pickle.load(open("model_pickle.dat", "rb"))

# %%
import pandas as pd
df = pd.read_csv('InputFormat.csv')

# %%
y_pred_proba = loaded_model.predict_proba(df) #Predicting probability
y_pred = loaded_model.predict(df) #Predicting Frauds 

# %%
df1= pd.DataFrame(y_pred_proba, columns = ['Non_Fraud_Probability','Fraud_Probabilty'])
df["Fraud"]= pd.DataFrame(y_pred, columns = ['Fraud'])
df["Fraud_Probabilty"] = df1["Fraud_Probabilty"]
df.head(10)