import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import numpy as np
import sklearn as skm

loan = pd.read_csv("loan_data.csv")

loan.drop(loan[(loan['person_age'] > 50) | (loan['person_emp_exp'] > 45)].index, inplace=True)

cat_var = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]

encoder = OneHotEncoder(sparse_output=False, drop="first")

cat_var_encoded = encoder.fit_transform(loan[cat_var])

column_names = encoder.get_feature_names_out(cat_var)
pd.DataFrame(cat_var_encoded, columns=column_names)
num_var = ["loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score", "loan_status"]

loan_encoded = pd.concat([
    pd.DataFrame(cat_var_encoded, columns=column_names, index=loan.index), 
    loan[num_var]
], axis=1)

#Now, we start the modeling 
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from collections import Counter
y = (loan_encoded['loan_status'] > 0).astype(int)

# Remove loan_status from features
X = loan_encoded.drop('loan_status', axis=1)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Standardize the numeric features
numeric_features = ["loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = StandardScaler()
smoteenn = SMOTEENN(sampling_strategy={
    0: int(sum(y_train == 0)),  # keep all majority class samples
    1: int(sum(y_train == 0) * 0.8)  # create minority samples up to 80% of majority
}, random_state=42)

X_train_scaled[numeric_features] = scaler.fit_transform(X_train_scaled[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test_scaled[numeric_features])

# Oversampling the training set
X_train_scaled_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scaled, y_train)

# Check the new distribution
print("Original distribution:", Counter(y_train))
print("Resampled distribution:", Counter(y_train_resampled))

# Create new balanced dataframe
loan_balanced = pd.concat([pd.DataFrame(X_train_scaled_resampled, columns=X_train.columns), 
                          pd.Series(y_train_resampled, name='loan_status')], axis=1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

Seq_model = Sequential()

Seq_model.add(Dense(128, input_dim=X_train_scaled_resampled.shape[1], activation='relu'))
Seq_model.add(Dropout(0.5))
Seq_model.add(Dense(64, activation='relu'))
Seq_model.add(Dropout(0.4))
Seq_model.add(Dense(32, activation='relu'))
Seq_model.add(Dropout(0.3))
Seq_model.add(Dense(16, activation='relu'))
Seq_model.add(Dropout(0.2))
Seq_model.add(Dense(8, activation='relu'))
Seq_model.add(Dropout(0.1))
Seq_model.add(Dense(1, activation='sigmoid'))

Seq_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = Seq_model.fit(X_train_scaled_resampled, y_train_resampled, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

from sklearn.metrics import roc_auc_score

y_pred_prob = Seq_model.predict(X_test_scaled).ravel()
auc_score = roc_auc_score(y_test, y_pred_prob)

print(f'ROC-AUC Score on Test Set: {auc_score:.4f}')

from joblib import dump
from tensorflow.keras.models import save_model

# 保存 scaler, encoder
dump(scaler, 'scaler_NN.pkl')
dump(encoder, 'encoder_NN.pkl')

# 保存模型
save_model(Seq_model, 'model_NN.h5')
