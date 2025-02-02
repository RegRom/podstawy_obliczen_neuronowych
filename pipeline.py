#%%

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler  

from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.python.keras.engine.training import Model

skf = StratifiedKFold(n_splits=5, random_state=444)


#Importy bibliotek do klasyfikacji
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from elm import ELM
# from autoencoder import AutoencoderClassifier

#Importy metod ekstrakcji
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Importy metryk
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

# %%
# Wygenerowanie zbioru danych

# X, y = make_classification(
#     n_features=1000, 
#     n_redundant=200,
#     n_informative=700,
#     n_clusters_per_class=1,
#     n_samples=1000,
#     weights=[ 0.5, 0.5 ]
#     )
# print(X.shape)
# print(y.shape)

# %%

ex1data = np.genfromtxt('datasets\\5000_features_2_classes_balanced.csv', delimiter=',')

# %%

X = ex1data[:, :-1]
y = ex1data[:, -1]
print(X.shape)
# %%
# Skalowanie danych i transpozycja kolumny z etykietami

y_T = y[:, np.newaxis]
print(y_T.shape)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# %%
#Wyodrębnianie zestawów cech i zapisanie do tablicy

feature_sets = [100, 200, 300, 500, 1000, 2500]
X_train_array = []

for set_size in feature_sets:
    X_set = X_scaled[:,:set_size]

    print(X_set.shape)

    X_train_array.append(X_set)

X_train_array.append(X_scaled)

# # %%
# test Extreme Learning Machine

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# elm_clf = ELM(X_train.shape[1], 1, 1000)

# elm_clf.train(X_train, y_train[:, np.newaxis])

# # %%

# elm_pred = elm_clf.predict(X_test)
# elm_pred = (elm_pred > 0.5).astype(int)

# print(round(balanced_accuracy_score(elm_pred, y_test), 2))

# # %%
# from tensorflow import keras

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# autoencoder = AutoencoderClassifier(X_train.shape[1])

# autoencoder.build_autoencoder_classifier()

# keras.utils.plot_model(autoencoder.autoencoder, "autoencoder.png", show_shapes=True)


# #%%

# def display_train_loop_results(scores_array):
#     headers = ['Algorithm', '1', '2', '3', '4', '5', '6']
#     scores_rows = [
#         ["SVC"] + scores_array[0],
#         ["kNN"] + scores_array[1],
#         ["GNB"] + scores_array[2],
#         ["DT"] + scores_array[3],
#         ["MLP"] + scores_array[4],
#         ["ELM"] + scores_array[5]
#     ]
#     scores_table = tabulate(scores_rows, headers=headers, tablefmt="pretty")
#     print(scores_table)

# %% Eksperyment bez ekstrakcji

def make_experiments_without_extract(X, y):
    svc_no_extract_scores = []
    knn_no_extract_scores = []
    gnb_no_extract_scores = []
    dt_no_extract_scores = []
    mlp_no_extract_scores = []
    elm_no_extract_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svc_clf = SVC(random_state=444)
        knn_clf = KNeighborsClassifier()
        gnb_clf = GaussianNB()
        dt_clf = DecisionTreeClassifier(random_state=444)
        mlp_clf = MLPClassifier(random_state=444)
        elm_clf = ELM(X_train.shape[1], 1, 1000)

        svc_pred = svc_clf.fit(X_train, y_train).predict(X_test)
        knn_pred = knn_clf.fit(X_train, y_train).predict(X_test)
        gnb_pred = gnb_clf.fit(X_train, y_train).predict(X_test)
        dt_pred = dt_clf.fit(X_train, y_train).predict(X_test)
        mlp_pred = mlp_clf.fit(X_train, y_train).predict(X_test)

        elm_clf.train(X_train, y_train[:, np.newaxis])
        elm_pred = elm_clf.predict(X_test)
        elm_pred = (elm_pred > 0.5).astype(int)

        svc_no_extract_scores.append(round(accuracy_score(svc_pred, y_test), 2))
        knn_no_extract_scores.append(round(accuracy_score(knn_pred, y_test), 2))
        gnb_no_extract_scores.append(round(accuracy_score(gnb_pred, y_test), 2))
        dt_no_extract_scores.append(round(accuracy_score(dt_pred, y_test), 2)) 
        mlp_no_extract_scores.append(round(accuracy_score(mlp_pred, y_test), 2))
        elm_no_extract_scores.append(round(accuracy_score(elm_pred, y_test), 2))
    
    return [
        round(np.average(svc_no_extract_scores), 2),
        round(np.average(knn_no_extract_scores), 2),
        round(np.average(gnb_no_extract_scores), 2),
        round(np.average(dt_no_extract_scores), 2),
        round(np.average(mlp_no_extract_scores), 2),
        round(np.average(elm_no_extract_scores), 2)
        ]

# # %%
# # Wyniki eksperymentu bez ekstrakcji

# display_train_loop_results(make_experiments_without_extract(X, y))

#%%
#Ekstrakcja danych z PCA
def make_experiments_with_pca(X, y):
    svc_pca_scores = []
    knn_pca_scores = []
    gnb_pca_scores = []
    dt_pca_scores = []
    mlp_pca_scores = []
    elm_pca_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pca = PCA()
        X_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        svc_clf = SVC(random_state=444)
        knn_clf = KNeighborsClassifier()
        gnb_clf = GaussianNB()
        dt_clf = DecisionTreeClassifier(random_state=444)
        mlp_clf = MLPClassifier(random_state=444)
        elm_clf = ELM(X_pca.shape[1], 1, 1000)

        svc_pred = svc_clf.fit(X_pca, y_train).predict(X_test_pca)
        knn_pred = knn_clf.fit(X_pca, y_train).predict(X_test_pca)
        gnb_pred = gnb_clf.fit(X_pca, y_train).predict(X_test_pca)
        dt_pred = dt_clf.fit(X_pca, y_train).predict(X_test_pca)
        mlp_pred = mlp_clf.fit(X_pca, y_train).predict(X_test_pca)

        elm_clf.train(X_pca, y_train[:, np.newaxis])
        elm_pred = elm_clf.predict(X_test_pca)
        elm_pred = (elm_pred > 0.5).astype(int)


        svc_pca_scores.append(round(accuracy_score(svc_pred, y_test), 2))
        knn_pca_scores.append(round(accuracy_score(knn_pred, y_test), 2))
        gnb_pca_scores.append(round(accuracy_score(gnb_pred, y_test), 2))
        dt_pca_scores.append(round(accuracy_score(dt_pred, y_test), 2)) 
        mlp_pca_scores.append(round(accuracy_score(mlp_pred, y_test), 2)) 
        elm_pca_scores.append(round(accuracy_score(elm_pred, y_test), 2)) 

    return [
        round(np.average(svc_pca_scores), 2),
        round(np.average(knn_pca_scores), 2),
        round(np.average(gnb_pca_scores), 2),
        round(np.average(dt_pca_scores), 2),
        round(np.average(mlp_pca_scores), 2),
        round(np.average(elm_pca_scores), 2)
        ]

# %%
# Wyniki eksperymentu z ekstrakcją poprzez PCA

# np.set_printoptions(suppress=True, precision=2)

# headers = ['Algorithm', '1', '2', '3', '4', '5']
# pca_scores_rows = [
#     ["SVC"] + svc_pca_scores,
#     ["kNN"] + knn_pca_scores,
#     ["GNB"] + gnb_pca_scores,
#     ["DT"] + dt_pca_scores,
#     ["MLP"] + mlp_pca_scores
# ]
# scores_table = tabulate(pca_scores_rows, headers=headers, tablefmt="pretty")
# print(scores_table)


#%%
#Ekstrakcja danych z kPCA

def make_experiments_with_kpca(X, y):
    svc_kpca_scores = []
    knn_kpca_scores = []
    gnb_kpca_scores = []
    dt_kpca_scores = []
    mlp_kpca_scores = []
    elm_kpca_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        kpca = KernelPCA()
        X_kpca = kpca.fit_transform(X_train)
        X_test_kpca = kpca.transform(X_test)

        svc_clf = SVC(random_state=444)
        knn_clf = KNeighborsClassifier()
        gnb_clf = GaussianNB()
        dt_clf = DecisionTreeClassifier(random_state=444)
        mlp_clf = MLPClassifier(random_state=444)
        elm_clf = ELM(X_kpca.shape[1], 1, 1000)

        svc_pred = svc_clf.fit(X_kpca, y_train).predict(X_test_kpca)
        knn_pred = knn_clf.fit(X_kpca, y_train).predict(X_test_kpca)
        gnb_pred = gnb_clf.fit(X_kpca, y_train).predict(X_test_kpca)
        dt_pred = dt_clf.fit(X_kpca, y_train).predict(X_test_kpca)
        mlp_pred = mlp_clf.fit(X_kpca, y_train).predict(X_test_kpca)

        elm_clf.train(X_kpca, y_train[:, np.newaxis])
        elm_pred = elm_clf.predict(X_test_kpca)
        elm_pred = (elm_pred > 0.5).astype(int)

        svc_kpca_scores.append(round(accuracy_score(svc_pred, y_test), 2))
        knn_kpca_scores.append(round(accuracy_score(knn_pred, y_test), 2))
        gnb_kpca_scores.append(round(accuracy_score(gnb_pred, y_test), 2))
        dt_kpca_scores.append(round(accuracy_score(dt_pred, y_test), 2)) 
        mlp_kpca_scores.append(round(accuracy_score(mlp_pred, y_test), 2))
        elm_kpca_scores.append(round(accuracy_score(elm_pred, y_test), 2)) 

    return [
        round(np.average(svc_kpca_scores), 2),
        round(np.average(knn_kpca_scores), 2),
        round(np.average(gnb_kpca_scores), 2),
        round(np.average(dt_kpca_scores), 2),
        round(np.average(mlp_kpca_scores), 2),
        round(np.average(elm_kpca_scores), 2)
        ]

# %%
# Wyniki eksperymentu z ekstrakcją poprzez kPCA

# np.set_printoptions(suppress=True, precision=2)

# headers = ['Algorithm', '1', '2', '3', '4', '5']
# kpca_scores_rows = [
#     ["SVC"] + svc_kpca_scores,
#     ["kNN"] + knn_kpca_scores,
#     ["GNB"] + gnb_kpca_scores,
#     ["DT"] + dt_kpca_scores,
#     ["MLP"] + mlp_kpca_scores
# ]
# scores_table = tabulate(kpca_scores_rows, headers=headers, tablefmt="pretty")
# print(scores_table)

#%%
#Ekstrakcja danych z LDA

svc_lda_scores = []
knn_lda_scores = []
gnb_lda_scores = []
dt_lda_scores = []
mlp_lda_scores = []
elm_lda_scores = []

def make_experiments_with_lda(X, y):
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)

        svc_clf = SVC(random_state=444)
        knn_clf = KNeighborsClassifier()
        gnb_clf = GaussianNB()
        dt_clf = DecisionTreeClassifier(random_state=444)
        mlp_clf = MLPClassifier(random_state=444)
        elm_clf = ELM(X_lda.shape[1], 1, 1000)

        svc_pred = svc_clf.fit(X_lda, y_train).predict(X_test_lda)
        knn_pred = knn_clf.fit(X_lda, y_train).predict(X_test_lda)
        gnb_pred = gnb_clf.fit(X_lda, y_train).predict(X_test_lda)
        dt_pred = dt_clf.fit(X_lda, y_train).predict(X_test_lda)
        mlp_pred = mlp_clf.fit(X_lda, y_train).predict(X_test_lda)

        elm_clf.train(X_lda, y_train[:, np.newaxis])
        elm_pred = elm_clf.predict(X_test_lda)
        elm_pred = (elm_pred > 0.5).astype(int)

        svc_lda_scores.append(round(accuracy_score(svc_pred, y_test), 2))
        knn_lda_scores.append(round(accuracy_score(knn_pred, y_test), 2))
        gnb_lda_scores.append(round(accuracy_score(gnb_pred, y_test), 2))
        dt_lda_scores.append(round(accuracy_score(dt_pred, y_test), 2)) 
        mlp_lda_scores.append(round(accuracy_score(mlp_pred, y_test), 2))
        elm_lda_scores.append(round(accuracy_score(elm_pred, y_test), 2)) 


    return [
        round(np.average(svc_lda_scores), 2),
        round(np.average(knn_lda_scores), 2),
        round(np.average(gnb_lda_scores), 2),
        round(np.average(dt_lda_scores), 2),
        round(np.average(mlp_lda_scores), 2),
        round(np.average(elm_lda_scores), 2)
        ] 


#%% 
def make_experiment_1(X, y):

    for X_train in X_train_array:
        no_extract = make_experiments_without_extract(X_train, y)
        pca = make_experiments_with_pca(X_train, y)
        kpca = make_experiments_with_kpca(X_train, y)
        lda = make_experiments_with_lda(X_train, y)

        feature_number = X_train.shape[1]

        headers = ['Algorithm', 'SVC', 'kNN', 'GNB', 'DT', 'MLP', 'ELM', 'Średnio']
        scores_rows = [
            [feature_number] + no_extract + [round(np.average(no_extract), 2)],
            ["PCA"] + pca + [round(np.average(pca), 2)],
            ["kPCA"] + kpca + [round(np.average(kpca), 2)],
            ["LDA"] + lda + [round(np.average(lda), 2)]
        ]

        SVC_avg = round(np.average([no_extract[0], pca[0], kpca[0], lda[0]]), 2)
        kNN_avg = round(np.average([no_extract[1], pca[1], kpca[1], lda[0]]), 2)
        GNB_avg = round(np.average([no_extract[2], pca[2], kpca[2], lda[2]]), 2)
        DT_avg = round(np.average([no_extract[3], pca[3], kpca[3], lda[3]]), 2)
        MLP_avg = round(np.average([no_extract[4], pca[4], kpca[4], lda[4]]), 2)
        ELM_avg = round(np.average([no_extract[5], pca[5], kpca[5], lda[5]]), 2)

        scores_rows.append(["Average", SVC_avg, kNN_avg, GNB_avg, DT_avg, MLP_avg, ELM_avg])

        scores_table = tabulate(scores_rows, headers=headers, tablefmt="latex")
        print(scores_table)

# %% Wykres algorytmów

# Data
df=pd.DataFrame({
        'Features': [100, 200, 300, 500, 1000, 2500, 5000],
        'SVC': [0.49 , 0.51, 0.48, 0.47 , 0.47 , 0.48 , 0.5 ], 
        'kNN': [0.47 , 0.48 , 0.48 , 0.47 , 0.52 , 0.6  , 0.68 ], 
        'GNB': [0.5, 0.54, 0.51 , 0.49 , 0.48 , 0.51 , 0.53 ], 
        'DT': [0.45 , 0.49 , 0.5  , 0.49 , 0.52 , 0.52 , 0.49 ],
        'MLP': [0.48 , 0.49 , 0.47 , 0.46 , 0.48 , 0.49 , 0.53 ],
        'ELM': [0.48 , 0.5  , 0.5  , 0.54 , 0.5  , 0.53 , 0.49 ],
    })

fig, ax = plt.subplots(figsize=(12,8))
# multiple line plot
fig.suptitle('Średnia wartość metryki Accuracy algorytmów w zależności od liczby cech', fontsize=16)
ax.set_xlabel('Liczba cech')
ax.set_ylabel('Wartość Accuracy')
plt.plot( 'Features', 'SVC', data=df, marker='o', color='blue', linestyle='dashed')
plt.plot( 'Features', 'kNN', data=df, marker='o', color='olive', linewidth=2)
plt.plot( 'Features', 'GNB', data=df, marker='o', color='orange', linewidth=2)
plt.plot( 'Features', 'DT', data=df, marker='o', color='green', linestyle='dashed')
plt.plot( 'Features', 'MLP', data=df, marker='o', color='black', linewidth=2, linestyle='dotted')
plt.plot( 'Features', 'ELM', data=df, marker='o', color='red', linewidth=2, linestyle='dotted')
plt.legend()
locs, labels = plt.xticks()
plt.xticks([100, 200, 300, 500, 1000, 2500, 5000], [100, 200, 300, 500, 1000, 2500, 5000], rotation='vertical')
plt.savefig('algo_avg.svg')
# %%
# URUCHOMIENIE EKSPERYMENTU 1
start = time.time()

make_experiment_1(X,y)

end = time.time()
duration = end - start
print(duration)

# %%

ex2data1 = np.genfromtxt('datasets\\1000_features_2_classes_balanced.csv', delimiter=',')
ex2data2 = np.genfromtxt('datasets\\1000_features_2_classes_0.33_0.67.csv', delimiter=',')
ex2data3 = np.genfromtxt('datasets\\1000_features_2_classes_0.25_0.75.csv', delimiter=',')
ex2data4 = np.genfromtxt('datasets\\1000_features_2_classes_0.2_0.8.csv', delimiter=',')
ex2data5 = np.genfromtxt('datasets\\1000_features_2_classes_0.1_0.9.csv', delimiter=',')



# %%

imbalanced_sets = [ex2data1, ex2data2, ex2data3, ex2data4, ex2data5]

def make_experiment_2(sets):
    imbalance_scores = []

    for feat_set in sets:
        X = feat_set[:, :-1]
        y = feat_set[:, -1]
        print(X.shape)
        scores = make_experiments_without_extract(X, y)
        imbalance_scores.append(scores)
    
    headers = ['Algorithm', 'SVC', 'kNN', 'GNB', 'DT', 'MLP', 'ELM']
    scores_rows = [
        ["1:1"] + imbalance_scores[0],
        ["1:2"] + imbalance_scores[1],
        ["1:3"] + imbalance_scores[2],
        ["1:4"] + imbalance_scores[3],
        ["1:9"] + imbalance_scores[4]
    ]

    scores_table = tabulate(scores_rows, headers=headers, tablefmt="latex")
    print(scores_table)

# %%

make_experiment_2(imbalanced_sets)

# %%

ex3data1 = np.genfromtxt('datasets\\1000_features_3_classes_balanced.csv', delimiter=',')
ex3data2 = np.genfromtxt('datasets\\1000_features_3_classes_0.25_0.25_0.5.csv', delimiter=',')
ex3data3 = np.genfromtxt('datasets\\1000_features_3_classes_0.2_0.2_0.6.csv', delimiter=',')
ex3data4 = np.genfromtxt('datasets\\1000_features_3_classes_0.1_0.1_0.8.csv', delimiter=',')
ex3data5 = np.genfromtxt('datasets\\1000_features_3_classes_0.05_0.05_0.9.csv', delimiter=',')

imbalanced_sets = [ex3data1, ex3data2, ex3data3, ex3data4, ex3data5]

# %%
    
def make_experiment_3(sets):
    imbalance_scores = []

    for feat_set in sets:
        X = feat_set[:, :-1]
        y = feat_set[:, -1]
        print(X.shape)
        scores = make_experiments_without_extract(X, y)
        imbalance_scores.append(scores)

    headers = ['Algorithm', 'SVC', 'kNN', 'GNB', 'DT', 'MLP', 'ELM']
    scores_rows = [
        ["1:1:1"] + imbalance_scores[0],
        ["1:1:2"] + imbalance_scores[1],
        ["1:1:3"] + imbalance_scores[2],
        ["1:1:8"] + imbalance_scores[3],
        ["1:1:18"] + imbalance_scores[4]
    ]

    scores_table = tabulate(scores_rows, headers=headers, tablefmt="latex")
    print(scores_table)

# %%
start = time.time()

make_experiment_3(imbalanced_sets)

end = time.time()
duration = end - start
print(duration)
# %%
