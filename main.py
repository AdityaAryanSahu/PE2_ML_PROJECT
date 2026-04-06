import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import time
import joblib

#visualising the performance of diff algos and their train times
def comparison():
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results_df['Model'], results_df['Train Time (s)'], color='skyblue')
    plt.title('Training Time per Model')
    plt.ylim(0,0.15)
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("train_time_comparison.png")
    plt.show()
    

    plt.figure(figsize=(8, 5))
    bars = plt.bar(results_df['Model'], results_df['Accuracy'], color='lightgreen')
    plt.title('Accuracy per Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.2)  
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")
    plt.show()
    
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results_df['Model'], results_df['features_used'], color='blue')
    plt.title('number of features used per Model')
    plt.ylabel('features used per Model')
    plt.ylim(0, 13)  
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("features_used.png")
    plt.show()
    

#loading the dataset
df = pd.read_csv("Students_Performance.csv")
print("dataset loaded successfully")

#dealing with missing data
df = df.drop('Student_ID', axis=1) #student id is dropped since its a string, doesnt help with infernce
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

feature_names = df_imputed.columns.tolist()

#label encoding using labelencoder
label_encoders = {}
for col in df_imputed.columns:
    try:
        df_imputed[col] = df_imputed[col].astype(float)
    except:
        le = LabelEncoder()
        df_imputed[col] = le.fit_transform(df_imputed[col].astype(str))
        label_encoders[col] = le
        
#create the feature array and label array
X = df_imputed.drop('Grade', axis=1)
y=df_imputed['Grade']


#split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42, stratify=y)

#sclaing the feature to a common scale using StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

#dict containing all the models to be tested
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'DecisionTree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'NaiveBayes': GaussianNB(),
}

results = []

for name, model in models.items():
  best_model = float('-inf')
  max_accuracy = float('-inf')
  max_f1 = float('-inf')
  max_train_time = float('-inf')
  features_used=float('-inf')
  for i in range(1,13):
    start = time.time()
    selector=SelectKBest(mutual_info_classif,k=i)
    selector.fit(X_train_scaled,y_train)
    x_train_new=selector.transform(X_train_scaled)
    x_test_new=selector.transform(X_test_scaled)
    model.fit(x_train_new, y_train)
    train_time = time.time() - start


    y_pred = model.predict(x_test_new)
    accuracy = round(accuracy_score(y_test, y_pred),3)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 3) 
    if f1>max_f1 and accuracy>max_accuracy:
      best_model = model
      max_accuracy=accuracy
      max_f1=f1
      max_train_time=train_time
      features_used=i

  results.append([name, best_model, max_train_time, max_accuracy, max_f1, features_used])


results_df = pd.DataFrame(results, columns=['Model', 'Model_Object','Train Time (s)', 'Accuracy', 'F1_Score','features_used'])
print(results_df)

comparison()

#saving all the models, encoders, selector , feature names and scaler for inference
for i, row in results_df.iterrows():
    name = row['Model']
    model = row['Model_Object']
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, filename)
    
joblib.dump(sc,'sc.pkl')
joblib.dump(selector,'selector.pkl')
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(feature_names, "feature_names.pkl")