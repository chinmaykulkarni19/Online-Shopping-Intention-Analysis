# Online Shopping Intention Analysis with Python

# I will start with this task by importing the necessary libraries and the data:
import scikitplot as skplt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
data = pd.read_csv(
    r"C:\Users\HP\Desktop\MACHINE PROJECT\online_shoppers_intention.csv")

# Now let’s have a look at the missing values and fill them by using the fillna method in Python pandas:

missing = data.isnull().sum()
print(missing)

data.fillna(0, inplace=True)

# Now have a look at product related bounce rates of customers:

x = data.iloc[:, [5, 6]].values
print(x.shape)

# Now let’s apply the K-elbow method to determine the number of clustering groups:


wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                max_iter=300,
                n_init=10,
                random_state=0,
                algorithm='full',
                tol=0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)

plt.rcParams['figure.figsize'] = (13, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize=20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# K Means Clustering

# According to the graph above, the maximum curvature is at the second index, that is, the number of optimal clustering groups for the duration of the product and the bounce rates is 2. Once the number of clusterings determined, we apply the K Means method and plot the clusters:

km = KMeans(n_clusters=2, init='k-means++',
            max_iter=300, n_init=10, random_state=0)
# get predicted cluster index for each sample: 0, 1, 2
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=50,
            c='yellow', label='Uninterested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1],
            s=50, c='pink', label='Target Customers')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
            :, 1], s=50, c='blue', label='centeroid')

plt.title('ProductRelated Duration vs Bounce Rate', fontsize=20)
plt.grid()
plt.xlabel('ProductRelated Duration')
plt.ylabel('Bounce Rates')
plt.legend()
plt.show()

# Looking at this K Means grouping plot, we can say with certainty that customers who spent more time on a product-related website are very less likely to leave the website after viewing a single page.

# Since K-Means is not a supervised learning method, we are adopting other ways of evaluating its clustering result. The leftmost column of the confusion matrix represents the actual label (True or False revenue), and the top row represents the expected clustering groups (uninterested customers or target customers):

le = LabelEncoder()
labels_true = le.fit_transform(data['Revenue'])

# get predicted clustering result label
labels_pred = y_means

# print adjusted rand index, which measures the similarity of the two assignments

score = metrics.adjusted_rand_score(labels_true, labels_pred)
print("Adjusted rand index: ")
print(score)

# print confusion matrix
#cm = metrics.plot_confusion_matrix(None, labels_true, labels_pred)
#print(cm)

plt_1 = skplt.metrics.plot_confusion_matrix(labels_true, labels_pred, normalize=False)
plt_2 = skplt.metrics.plot_confusion_matrix(labels_true, labels_pred, normalize=True)
