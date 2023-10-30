#%% 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

base_data = pd.read_csv("../../tests/testdata/iris_with_spaces_full.csv")
labels = base_data['Species'].copy()
base_data.drop(columns=['Species'], inplace=True)
base_data.head()
# %%
kmeans = KMeans(init="random", n_clusters=3, random_state=0)
estimator = make_pipeline(StandardScaler(), kmeans)
estimator.fit(base_data)



# %%
silhouette_score = metrics.silhouette_score(
            base_data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
print(f'''
      
      Clustering Complete: Silhouette Score is {silhouette_score.round(3)}
      ''')
# %%
import pickle
print('''Saving pipeline to pickel file to "model.pkl" ''')
with open("model.pkl", 'wb') as pklfile:
    pickle.dump(estimator, pklfile)


# %%
