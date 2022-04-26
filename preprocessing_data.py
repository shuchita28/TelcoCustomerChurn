import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
import math
import pickle

# Read the dataset
df = pd.read_excel('/Users/shuchitamishra/Desktop/DS5220 SML/Project/Telco_customer.xlsx')

# Set the target variable
targetVariableName = 'Churn Label'
df_model = df.copy()
del df_model['Churn Reason']

'''
Handle missing values: Replace empty string with NULL-
this is because Total charges has 7 rows with empty spaces
'''
df_model.replace(r'^\s*$', np.nan, regex=True, inplace=True)


def imputeTotalCharges(df):
    for index, row in df.iterrows():
        if(math.isnan(row['Total Charges'])):
            df.at[index, 'Total Charges'] = df.at[index, 'Monthly Charges'] * df.at[index, 'Tenure Months']
    return df


df_model = imputeTotalCharges(df_model)
df_model['Total Charges'] = pd.to_numeric(df_model['Total Charges'], errors='coerce')
#print(df_model['Total Charges'].dtype)

'''Feature engineering		
Now lets drop unnecessary columns that we know are of no use
Dropping Country columns as it has only one value type - United States.
Also only California state is present Other columns we will drop are
State, CustomerID,City,ZipCode, Count,Lat Long, Latitude,Longitude,Churn Score,
CLTV, Churn Reason. Dropping Country columns as it has only one value type -
United States. Also only California state is present'''


del df_model['Country']
del df_model['State']
del df_model['CustomerID']
del df_model['City']
del df_model['Zip Code']


# Dropping Count columns as it has only one value type - 1
print(df_model['Count'].value_counts())
del df_model['Count']

# Also deleting latitute and longitutes as it has no purpose and other target related variable
for items in ['Lat Long', 'Latitude', 'Longitude', 'Churn Score', 'CLTV']:
    del df_model[items]




df_2 = df_model.copy()
codes = {'Male': 0, 'Female': 1}
df_2['Gender'] = df_2['Gender'].map(codes)

codes = {'No': 0, 'Yes': 1}
for items in ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']:
    df_2[items] = df_2[items].map(codes)

codes = {'No': 0, 'Yes': 1, 'No phone service': 2}
df_2['Multiple Lines'] = df_2['Multiple Lines'].map(codes)

codes = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
df_2['Internet Service'] = df_2['Internet Service'].map(codes)

codes = {'Yes': 0, 'No': 1, 'No internet service': 2}
for items in ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']:
    df_2[items] = df_2[items].map(codes)

codes = {'Month-to-month': 0, 'Two year': 1, 'One year': 2}
df_2['Contract'] = df_2['Contract'].map(codes)

codes = {'Mailed check': 0, 'Electronic check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
df_2['Payment Method'] = df_2['Payment Method'].map(codes)

x_ols_features = ['Gender', 'Senior Citizen', 'Partner', 'Dependents',
                  'Tenure Months', 'Phone Service', 'Multiple Lines',
                  'Internet Service', 'Online Security', 'Online Backup',
                  'Device Protection', 'Tech Support', 'Streaming TV',
                  'Streaming Movies', 'Contract', 'Paperless Billing',
                  'Payment Method', 'Monthly Charges', 'Total Charges']
y_ols = df_2["Churn Value"]


def get_stats():
    x_ols = df_2[x_ols_features]
    results = sm.OLS(y_ols, x_ols.astype(float)).fit()
    results_summary = results.summary()
    results_as_html = results_summary.tables[1].as_html()
    result_sum = pd.read_html(results_as_html, header=0, index_col=0)[0]
    p_val = result_sum['P>|t|']
    if max(p_val) <= 0.05:
        return x_ols_features, p_val, False
    m = 'Column to be removed is ' + str(p_val.idxmax())

    x_ols_features.remove(p_val.idxmax())
    return x_ols_features, p_val, True


con = True
while(con):
    x_ols_features, p_val, con = get_stats()

df_model = df_model[x_ols_features]

# Get list of all categorical variables
cat_columns = [cname for cname in df_model.columns if df_model[cname].dtype == "object"]
#print(cat_columns)

# OneHotEncode the categorical variables
encoder = OneHotEncoder(sparse=False)
train_X_encoded = pd.DataFrame(encoder.fit_transform(df_model[cat_columns]))
train_X_encoded.columns = encoder.get_feature_names(cat_columns)
df_model.drop(cat_columns, axis=1, inplace=True)
df_model2 = pd.concat([df_model, train_X_encoded], axis=1)
df_model2[targetVariableName] = y_ols

# Label Encode the target variable
number = LabelEncoder()
df_model2[targetVariableName] = number.fit_transform(df_model2[targetVariableName])
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

df_model2.to_csv('FinalProcessed.csv')
