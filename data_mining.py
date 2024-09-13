import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# classification libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# clustering libraries
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def insert_data():
    folder = 'harth'
    df_list = []
    counter = 1 
    for f in os.listdir(folder):
        file = os.path.join(folder, f)
        df_temp = pd.read_csv(file)
        row, _ = df_temp.shape
        df_temp.insert(1, "userID", row*[counter])
        counter = counter + 1
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    df = df.drop(columns=['index', 'Unnamed: 0'])

    def data_infos():
            #1 df info
            print(df.info())
            #2 check about empty data
            NaNs_data_df =  df.isnull().sum()
            print("\nCheck empty data:")
            print(NaNs_data_df)
            #3 df statistics
            df_stats = df.describe()
            df_stats.to_csv('describe.csv')
            #4 create diagrams
            diagrams(df_stats)

    def diagrams(df_stats):
        hist_df = df.drop(['timestamp'], axis = 1)
        hist_df.hist(bins=80, layout=(3, 3), figsize=(15, 10), color='skyblue', edgecolor='black')
        plt.savefig('figures/histograms.png')
        plt.close()

        df['label'].value_counts().plot(kind='bar')
        plt.title('Frequency of Labels')
        plt.xlabel('Label')
        plt.ylabel('Frequency')
        plt.savefig('figures/labels_frequency.png')
        plt.close()

        corr_df = df.drop(['timestamp'], axis = 1)
        sns.heatmap(corr_df.corr(),annot=True)
        plt.savefig('figures/correlogram1.png')
        plt.close()

        selected_stats = df_stats.drop(index='count')
        sns.heatmap(selected_stats.transpose().corr(),annot=True)
        plt.savefig('figures/correlogram2.png')
        plt.close()

    data_infos()

    return df
    

def classification(df, option):
    cl_dict = {1: 'RandomForestClassifier', 2: 'MLPClassifier', 3: 'GaussianNB'}

    def select_classifier(option=None):
        if option == 1:
            classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
        elif option == 2:
            classifier = MLPClassifier(max_iter=10000, activation='logistic', random_state=42)
        elif option == 3:
            classifier = GaussianNB()
        else:
            raise ValueError("option values are : 1,2,3")
        return classifier

    def run_classification(X,Y,classifier = 1):
        X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=42)
        #training
        model = select_classifier(classifier)
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        print("\n---------------------------------------------------------------------")
        print(f"Classifier {cl_dict[classifier]} yeilds training accuracy of {model.score(X_train,y_train)}\nwith a testing accuracy of {accuracy_score(y_test, predictions)}")
        return cl_dict[classifier],model.score(X_train,y_train),accuracy_score(y_test, predictions)

    X = df.drop(['label', 'timestamp', 'userID'],axis = 1)
    Y = df['label']

    models_train_acc = []
    models_test_acc = []
    cl_model = []

    if option == None:
        for i in range (1,4):
            classifier, train_acc, test_acc = run_classification(X,Y,i)
            models_train_acc.append(train_acc)
            models_test_acc.append(test_acc)
            cl_model.append(classifier)

        plt.figure("Classification Results")
        x_axis = np.arange(len(cl_model))
        plt.bar(x_axis-0.2,models_train_acc,0.4,label = "Train set")
        plt.bar(x_axis+0.2,models_test_acc,0.4,label = 'Test Set')
        plt.xticks(x_axis,cl_model)
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('figures/Classification_results.png')
    else:
        run_classification(X,Y,option)

def clustering(X, option=None):
    def kmeans():
        # find best k value
        kmeans = KMeans(n_init=10)
        vis = KElbowVisualizer(kmeans,numeric_only=None)
        vis.fit(X)
        vis.show(outpath="figures/kmeans_kelbow.png")
        best_k = vis.elbow_value_
        # kmeans
        labels = KMeans(n_clusters=best_k).fit(X).labels_
        print('\n*** KMEANS ***')
        print(f'Silhouette Score: {silhouette_score(X, labels)}')
        print(f'Labels: {labels}')
        print(f'Number of clusters: {best_k}')

    
    def meanshift():
        labels = MeanShift().fit(X).labels_
        print('\n*** MEANSHIFT ***')
        print(f'Silhouette Score: {silhouette_score(X, labels)}')
        print(f'Labels: {labels}')
        print(f'Number of clusters: {len(np.unique(labels))}')

    def dbscan():
        # find the optimal eps value
        nn = NearestNeighbors(n_neighbors=5).fit(X)
        distances, _ = nn.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.figure(figsize=(10,8))
        plt.plot(distances)
        plt.savefig('figures/5-NN.png')
        # dbscan
        labels = DBSCAN(eps=42500,min_samples=2).fit(X).labels_
        print('\n*** DBSCAN ***')
        print(f'Silhouette Score: {silhouette_score(X, labels)}')
        print(f'Labels: {labels}')
        print(f'Number of clusters: {len(np.unique(labels))}')

    def select_clusterer(option):
        if option == 1:
            kmeans()
        elif option == 2:
            meanshift()
        elif option == 3:
            dbscan()
        else:
            raise ValueError("option values are : 1, 2, 3")

    if option == None:
        for i in range(1,4):
            select_clusterer(i)
    else:
        select_clusterer(option)

def dataframe_processor(df):
    df_list = []
    for i in range(1,23):
        df_user = df.loc[df['userID'] == i]
        df_temp = df_user.drop(columns=['timestamp', 'label', 'userID']).describe()
        df_temp = df_temp.transpose()
        flat_df = pd.DataFrame([df_temp.values.flatten()])
        df_list.append(flat_df)

    return pd.concat(df_list, ignore_index=True)

if __name__ == "__main__":
    try:  
        os.mkdir('figures/')  
    except OSError as error:  
        print('The directory has already been created.',error) 

    df = insert_data()
    # df = pd.read_csv('harth/S006.csv')

    # # Matching: {'RandomForestClassifier': 1, 'MLPClassifier': 2, 'GaussianNB': 3}
    classification(df)

    X = dataframe_processor(df)
    # # Matching: {'kmeans': 1, 'meanshift': 2, 'dbscan': 3}
    clustering(X)
