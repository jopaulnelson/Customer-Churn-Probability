import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

#Load data
df= pd.read_csv(r'E:\Jopaul\Prject app\European_Bank.csv')

#Feature Engineering
df['BalanceSalaryRatio']=df['Balance']/(df['EstimatedSalary']+1)
df['ProductDensity']=df['NumOfProducts']/(df['Tenure']+1)
df['EngagementScore']=df['IsActiveMember']*df['HasCrCard']
df['AgeTenure']=df['Age']*df['Tenure']

#Drop columns
df.drop(['CustomerId','Surname'], axis=1, inplace=True)

#Encoding
df=pd.get_dummies(df, drop_first=True)
x=df.drop('Exited', axis=1)
y=df['Exited']

#Scaling
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_train, x_test, y_train, y_test= train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

#k-fold Cross Validation
model= LogisticRegression(max_iter=1000)
scores= cross_val_score(model, x_scaled, y, cv=5)
print("Cross-validation scores:", scores)

#LogisticRegression
model_lr= LogisticRegression(max_iter=1000)
model_lr.fit(x_train, y_train)
accuracy_lr= model_lr.score(x_test, y_test)
print("Logistic Regression Accuracy:", accuracy_lr)

#DecisionTree
model_dt= DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
accuracy_dt= model_dt.score(x_test, y_test)
print("Decision Tree Accuracy:", accuracy_dt)

#RandomForest
model_rf= RandomForestClassifier(class_weight='balanced')
model_rf.fit(x_train, y_train)
accuracy_rf= model_rf.score(x_test, y_test)
print("Random Forest Accuracy:", accuracy_rf)

#Model Evaluation
y_pred= model_rf.predict(x_test)
y_prob= model_rf.predict_proba(x_test)[:,1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nRoc_Auc:", roc_auc_score(y_test, y_prob))

#Model Explainability
importances= model_rf.feature_importances_
feature_names= x.columns
importance_df= pd.DataFrame({'Feature': feature_names, 'Importance': importances
                             }).sort_values(by='Importance', ascending=False)
print(importance_df.head(10))

#Predictive Target
results= pd.DataFrame({'Actual': y_test, 'Predicted':y_pred, 'Churn Probability':y_prob})

def risk_category(p):
  if p<0.3:
    return 'Loww Risk'
  if p<0.7:
    return 'Medium Risk'
  else:
    return 'High Risk'
results["risk_category"]= results["Churn Probability"].apply(risk_category)
print(results.head())

results['risk_category'].value_counts()
results.sort_values(by='Churn Probability', ascending=False).head(10)
results.groupby('risk_category')['Churn Probability'].mean()

results['Age'] = x_test[:, list(x.columns).index('Age')]
results.groupby('risk_category')['Age'].mean()
print(results.head())

# Save model
joblib.dump(model_rf, open("model.pkl", "wb"))

# Save scaler
joblib.dump(scaler, open("scaler.pkl", "wb"))

# Save feature columns
joblib.dump(x.columns.tolist(), open("features.pkl", "wb"))
