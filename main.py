# Imports 
import pandas as pd
import numpy as np
import re

# Text Vectorization, Data Splitting, & Performance Metrics
# Converts raw text into a numerical TF-IDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Divides the dataset into training and testing subsets.
from sklearn.model_selection import train_test_split

# Functions to evaluate model performance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc)


# Models
# Baseline model used in lab1
from sklearn.naive_bayes import GaussianNB 

# Models for proposed improvement
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Statistical tests
from scipy.stats import wilcoxon


# # Text Cleaning & Stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 

def remove_html(text):
    return re.compile(r'<.*?>').sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

stop_words = stopwords.words('english') + ['...']

def remove_stopwords(text):
    return " ".join([w for w in str(text).split() if w not in stop_words])

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[\\'\"]", "", text)
    return text.strip().lower()

def preprocess(text):
    """Run all cleaning steps in one call."""
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = clean_str(text)
    return text

# Cliff's Delta 
# Measures how large the difference between two models is.
def cliffs_delta(a, b):
    a, b = np.array(a), np.array(b)
    more = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    d = (more - less) / (len(a) * len(b))
    if abs(d) < 0.147:   size = "negligible"
    elif abs(d) < 0.33:  size = "small"
    elif abs(d) < 0.474: size = "medium"
    else:                size = "large"
    return d, size

# Projects list and REPEAT
projects = ['caffe', 'keras', 'incubator-mxnet', 'pytorch', 'tensorflow']
REPEAT = 30

for project in projects:
    print(f"\n{'='*60}")
    print(f"PROJECT: {project.upper()}")
    print(f"{'='*60}")

    # Loading and preparing data 
    df = pd.read_csv(f'data/{project}.csv')
    df['text'] = df.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )
    df['text'] = df['text'].apply(preprocess)
    df = df.rename(columns={"class": "sentiment"}).fillna('')

    # Empty lists for storing results 
    nb_acc,  nb_prec,  nb_rec,  nb_f1_mac,  nb_f1_bin,  nb_auc  = [], [], [], [], [], []
    svm_acc, svm_prec, svm_rec, svm_f1_mac, svm_f1_bin, svm_auc = [], [], [], [], [], []
    rf_acc,  rf_prec,  rf_rec,  rf_f1_mac,  rf_f1_bin,  rf_auc  = [], [], [], [], [], []

    for i in range(REPEAT):

        # Train/test split (70/30, stratified) 
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['sentiment'],
            test_size=0.3,
            random_state=i,
            stratify=df['sentiment']
        )

        # TF-IDF vectorization 
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True)
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec  = tfidf.transform(X_test)

        # Baseline: Gaussian Naive Bayes 
        nb = GaussianNB()
        nb.fit(X_train_vec.toarray(), y_train)
        nb_pred   = nb.predict(X_test_vec.toarray())
        nb_scores = nb.predict_proba(X_test_vec.toarray())[:, 1]

        nb_acc.append(accuracy_score(y_test, nb_pred))
        nb_prec.append(precision_score(y_test, nb_pred, average='macro', zero_division=0))
        nb_rec.append(recall_score(y_test, nb_pred, average='macro', zero_division=0))
        nb_f1_mac.append(f1_score(y_test, nb_pred, average='macro', zero_division=0))
        nb_f1_bin.append(f1_score(y_test, nb_pred, pos_label=1, zero_division=0))
        fpr, tpr, _ = roc_curve(y_test, nb_scores, pos_label=1)
        nb_auc.append(auc(fpr, tpr))

        # Proposed Model 1: Linear SVM 
        svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
        svm.fit(X_train_vec, y_train)
        svm_pred   = svm.predict(X_test_vec)
        svm_scores = svm.decision_function(X_test_vec)

        svm_acc.append(accuracy_score(y_test, svm_pred))
        svm_prec.append(precision_score(y_test, svm_pred, average='macro', zero_division=0))
        svm_rec.append(recall_score(y_test, svm_pred, average='macro', zero_division=0))
        svm_f1_mac.append(f1_score(y_test, svm_pred, average='macro', zero_division=0))
        svm_f1_bin.append(f1_score(y_test, svm_pred, pos_label=1, zero_division=0))
        fpr, tpr, _ = roc_curve(y_test, svm_scores, pos_label=1)
        svm_auc.append(auc(fpr, tpr))

        # Proposed Model 2: Random Forest 
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train_vec, y_train)
        rf_pred   = rf.predict(X_test_vec)
        rf_scores = rf.predict_proba(X_test_vec)[:, 1]

        rf_acc.append(accuracy_score(y_test, rf_pred))
        rf_prec.append(precision_score(y_test, rf_pred, average='macro', zero_division=0))
        rf_rec.append(recall_score(y_test, rf_pred, average='macro', zero_division=0))
        rf_f1_mac.append(f1_score(y_test, rf_pred, average='macro', zero_division=0))
        rf_f1_bin.append(f1_score(y_test, rf_pred, pos_label=1, zero_division=0))
        fpr, tpr, _ = roc_curve(y_test, rf_scores, pos_label=1)
        rf_auc.append(auc(fpr, tpr))

    # --- Results Table ---
    print(f"\n{'Model':<12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1-Mac':>8} {'F1-Bin':>8} {'AUC':>7}")
    print("-" * 58)

    print(f"{'NB (Base)':<12} "
          f"{np.mean(nb_acc):>7.4f} "
          f"{np.mean(nb_prec):>7.4f} "
          f"{np.mean(nb_rec):>7.4f} "
          f"{np.mean(nb_f1_mac):>8.4f} "
          f"{np.mean(nb_f1_bin):>8.4f} "
          f"{np.mean(nb_auc):>7.4f}")

    print(f"{'LinearSVM':<12} "
          f"{np.mean(svm_acc):>7.4f} "
          f"{np.mean(svm_prec):>7.4f} "
          f"{np.mean(svm_rec):>7.4f} "
          f"{np.mean(svm_f1_mac):>8.4f} "
          f"{np.mean(svm_f1_bin):>8.4f} "
          f"{np.mean(svm_auc):>7.4f}")

    print(f"{'Rnd Forest':<12} "
          f"{np.mean(rf_acc):>7.4f} "
          f"{np.mean(rf_prec):>7.4f} "
          f"{np.mean(rf_rec):>7.4f} "
          f"{np.mean(rf_f1_mac):>8.4f} "
          f"{np.mean(rf_f1_bin):>8.4f} "
          f"{np.mean(rf_auc):>7.4f}")

    # Statistical Tests 
    print(f"\n--- Statistical Analysis vs Baseline ---")

    # SVM vs Baseline
    _, p_svm_mac = wilcoxon(nb_f1_mac, svm_f1_mac)
    d_svm_mac, size_svm_mac = cliffs_delta(svm_f1_mac, nb_f1_mac)
    print(f"  SVM vs NB (F1-Macro):  p={p_svm_mac:.4f}, Cliff's d={d_svm_mac:.3f} ({size_svm_mac})")

    _, p_svm_bin = wilcoxon(nb_f1_bin, svm_f1_bin)
    d_svm_bin, size_svm_bin = cliffs_delta(svm_f1_bin, nb_f1_bin)
    print(f"  SVM vs NB (F1-Binary): p={p_svm_bin:.4f}, Cliff's d={d_svm_bin:.3f} ({size_svm_bin})")

    # RF vs Baseline
    _, p_rf_mac = wilcoxon(nb_f1_mac, rf_f1_mac)
    d_rf_mac, size_rf_mac = cliffs_delta(rf_f1_mac, nb_f1_mac)
    print(f"  RF  vs NB (F1-Macro):  p={p_rf_mac:.4f}, Cliff's d={d_rf_mac:.3f} ({size_rf_mac})")

    _, p_rf_bin = wilcoxon(nb_f1_bin, rf_f1_bin)
    d_rf_bin, size_rf_bin = cliffs_delta(rf_f1_bin, nb_f1_bin)
    print(f"  RF  vs NB (F1-Binary): p={p_rf_bin:.4f}, Cliff's d={d_rf_bin:.3f} ({size_rf_bin})")