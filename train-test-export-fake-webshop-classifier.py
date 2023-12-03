"""
This script produces a classsifier of fake webshops. 
To do this, it injests processed data, trains and tests a classifer, and outputs the pickled classifier. 

The classifier is a Pipeline() containing a TfidfVectorizer() and a RandomForestClassifier()

In more detail - it imports data from WebScan, combines it, and performs analysis to identify fake webshops. The expected
inputs include data files in pickle format and a CSV file containing confirmed fake webshops. It generates
a model pipeline using RandomForestClassifier to classify webshops as 'Yes' (fake) or 'No' (not fake) based
on textual and domain features.

Expected Inputs:
- Pickle files with webshop data
- CSV file ('df_fake_domains.csv') containing confirmed fake webshops

Expected Output:
- Trained pipeline ('fake-webshop-pipeline-[date].pkl')
- Evaluation metrics such as confusion matrix, precision, recall, accuracy, F1 score

Dependencies:
- pandas, nltk, scikit-learn
"""

import os
import glob
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from typing import Dict, List, Any

# Initialize an empty dictionary to store data
data: Dict[str, Any] = {
    'DATA_DIR': str,
    'SEL_DATE': '2018-12-14', # Selected data
    'SPLIT_SEED': 2,
    'SPLIT_TEST_SIZE': 0.5,
    'clf': RandomForestClassifier,
    'd': pd.DataFrame,
    'data_files': str,
    'df': pd.DataFrame,
    'df_fake_domains': pd.DataFrame,
    'dfs': list,
    'ls_custom_stop': list,
    'ls_custom_stop_set': list,
    'ls_fake_domains': list,
    'tokenizer': RegexpTokenizer,
    'vec_tfidf': TfidfVectorizer
}

# Define data directory
data['DATA_DIR'] = os.path.join(os.environ['HOME'], f"dns-flag-day-{data['SEL_DATE']}-content")

# Import data from WebScan
data['data_files'] = glob.glob(os.path.join(data['DATA_DIR'], '*.p'))
data['dfs'] = [pd.read_pickle(file) for file in data['data_files']]
data['d'] = pd.concat(data['dfs'])

# Import list of confirmed fake webshops
data['df_fake_domains'] = pd.read_csv('df_fake_domains.csv')
data['ls_fake_domains'] = data['df_fake_domains']['domains'].values.tolist()

# Group the text by domain
data['df'] = data['d'][['domain', 'BodyText']].groupby('domain')['BodyText'].apply(lambda x: " ".join(x)).reset_index()
data['df']['label'] = data['df']['domain'].apply(lambda x: 'Yes' if x in data['ls_fake_domains'] else 'No')

# Define the tokenizer
data['ls_custom_stop'] = stopwords.words('english') + ['nz', 'st', 'www', 'co', 'new', 'zealand', 'nzd', 'us', 'ml', 'javascript']
data['ls_custom_stop_set'] = set(data['ls_custom_stop'])
data['tokenizer'] = RegexpTokenizer(r'[a-zA-Z]{2,}')

def gen_tokens(text):
    return [w.lower() for w in data['tokenizer'].tokenize(text) if w not in data['ls_custom_stop_set']]

# Define classifier components
clf_steps = [
    (data['vec_tfidf'], TfidfVectorizer(stop_words=data['ls_custom_stop'],
                                   tokenizer=gen_tokens,
                                   max_features=1000,
                                   ngram_range=(1, 2))),
    (data['clf'], RandomForestClassifier(random_state=42,
                                   max_features='auto',
                                   n_estimators=200))
]

# Construct the pipeline
data['clf'] = Pipeline(clf_steps)

# Create train and test datasets
x_train, x_test, y_train, y_test = train_test_split(
    data['df']['BodyText'],
    data['df']['label'],
    test_size=data['SPLIT_TEST_SIZE'],
    random_state=data['SPLIT_SEED'])

# Train the pipeline
data['clf'].fit(x_train, y_train)

# Make predictions to test the pipeline
y_pred = data['clf'].predict(x_test).tolist()

# Create a confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Print key experimental results
print(cnf_matrix)
print("Training df length: ", len(y_train))

print('Test Recall - % of labelled positives that are correctly predicted')
print("%.3f" % (tp / (tp + fn)))

print('Test Precision - % of predicted positives are actual positives')
print("%.3f" % (tp / (tp + fp)))

print('Test Accuracy - % of predictions that are actually correct')
print("%.3f" % ((tp + tn) / (tn + fn + tp + fp)))

print('F1 Score:')
print("%.3f" % (f1_score(y_test, y_pred, average="macro")))

# Check the k-fold cross-validation scores
scores = cross_val_score(data['clf'], x_train, y_train, cv=10, scoring='accuracy')
print(scores)

# Export the trained pipeline
joblib.dump(data['clf'], 'fake-webshop-pipeline-[date].pkl')
