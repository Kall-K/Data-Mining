import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
# classification libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# clustering libraries
from sklearn.cluster import KMeans

# timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label

def insert_data():
    folder = 'harth'
    df_list = []
    # counter = 1 
    for f in os.listdir(folder):
        file = os.path.join(folder, f)
        df_temp = pd.read_csv(file)
        # row, _ = df_temp.shape
        # df_temp.insert(1, "userID", row*[counter])
        # counter = counter + 1
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    df = df.drop(columns=['index', 'Unnamed: 0'])

    #-------------------------------------------
    #1 df info
    print(df.info())
    #2 check about empty data
    NaNs_data_df =  df.isnull().sum()
    print("Check empty data:")
    print(NaNs_data_df)
    #3 df statistics
    df_stats = df.describe()
    df_stats.to_csv('descride.csv', index=False)
    #4 create diagrams
    # diagrams(df, df_stats) #remove comment to create diagrams
    #-------------------------------------------

    return df
    # process_dataframe(df)

    # hist_df = df.drop(['timestamp'], axis = 1)
    # hist_df.hist(bins=80, layout=(3, 3), figsize=(15, 10), color='skyblue', edgecolor='black')
    # plt.savefig('Histogram.png')

    # corr_df = df.drop(['timestamp'], axis = 1)
    # sns.heatmap(corr_df.corr(),annot=True)
    # plt.savefig('Heatmap.png')

def diagrams(df, df_stats):
    try:  
        os.mkdir('figures/')  
    except OSError as error:  
        print('The directory has already been created.',error)  

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

    transposed_df = df_stats.T
    selected_stats = transposed_df.drop(['count'], axis = 1)
    sns.heatmap(selected_stats.corr(),annot=True)
    plt.savefig('figures/correlogram1.png')

#
#   ..............
#

def classification(df):
    cl_dict = {1: 'RandomForestClassifier', 2: 'MLPClassifier', 3: 'GaussianNB'}

    def select_classifier(option=1):
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
        print("---------------------------------------------------------------------")
        print(f"Classifier {cl_dict[classifier]} yeilds training accuracy of {model.score(X_train,y_train)}\nwith a testing accuracy of {accuracy_score(y_test, predictions)}")
        return cl_dict[classifier],model.score(X_train,y_train),accuracy_score(y_test, predictions)

    X = df.drop(['label', 'timestamp'],axis = 1)
    Y = df['label']

    models_train_acc = []
    models_test_acc = []
    cl_model = []

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
    plt.savefig('Classification_results.png')


def clustering(df):

    

    X = df
    kmeans = KMeans(n_clusters=12)
    kmeans.fit(X)

    labels = kmeans.labels_
    print(labels)

    centroids = kmeans.cluster_centers_
    # print(centroids)
    
def process_dataframe(df):
    df_list = []
    df = df.drop(columns=['timestamp', 'label'])
    df_cp = df.drop(columns=['userID'])
    df.rename(columns={'back_x': 'm_back_x','back_y': 'm_back_y','back_z': 'm_back_z','thigh_x': 'm_thigh_x','thigh_y': 'm_thigh_y','thigh_z': 'm_thigh_z'}, inplace=True) 
    df_cp.rename(columns={'back_x': 's_back_x','back_y': 's_back_y','back_z': 's_back_z','thigh_x': 's_thigh_x','thigh_y': 's_thigh_y','thigh_z': 's_thigh_z'}, inplace=True)
    
    for i in range(1,23):
        f_df_m = df.loc[df['userID'] == i] 
        f_df_std = df_cp.loc[df['userID'] == i] 

        m_df = f_df_m.mean(axis=0)
        std_df = f_df_std.std(axis=0)
 
        df_temp = pd.concat([m_df.to_frame().transpose(), std_df.to_frame().transpose()], axis=1, join="inner") 

        # print(df_temp.head())
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    print(df.head(23))

    X = df.drop(columns=['userID'])
    # print(X)
    # distortion = []
    # for k in range(2,10):
    #     kmeans = KMeans(n_clusters=k,n_init=10)
    #     kmeans.fit(X)
    #     distortion.append(kmeans.inertia_)

    # plt.plot(range(2, 10), distortion, marker = 'x' )
    # plt.xlabel('Number of clusters')
    # plt.ylabel('SSE')
    # plt.show()


    kmeans = KMeans(n_clusters=4)
    sil_vis = KElbowVisualizer(kmeans, numeric_only=None)

    sil_vis.fit(X)
    sil_vis.show()
    print("1:",kmeans.labels_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_init=10)
    sil_vis = KElbowVisualizer(kmeans,numeric_only=None)
    sil_vis.fit(X_scaled)
    sil_vis.show()
    print("2:",kmeans.labels_)
    # return df

if __name__ == "__main__":
    # df = insert_data()
    df = pd.read_csv('harth/S006.csv')
    classification(df)

    # clustering()
