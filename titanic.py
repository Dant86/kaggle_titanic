import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

"""
    Initialize dataframes
"""
path_to_data = "../data_dump/titanic/"
train = pd.read_csv(path_to_data + "train.csv")
test = pd.read_csv(path_to_data + "test.csv")

"""
    Clean train and test data; replace qualitative
    with quanititative data
"""
train["Sex"] = train["Sex"].map({'female': 1, 'male': 0})
train["Embarked"] = train["Embarked"].map({'Q': 0, 'C': 1, 'S': 2})
test["Sex"] = test["Sex"].map({'female': 1, 'male': 0})
test["Embarked"] = test["Embarked"].map({'Q': 0, 'C': 1, 'S': 2})

"""
    Isolate all columns without NaN for PCA; turn them into
    feature vectors with the following features:
    0: sex
    1: age
    2: sibSp
    3: parch
    4: fare
    5: embarked
"""
selected_features = ["Sex", "Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked"]
cleaned_train = train.dropna(axis=0, how="any")
cleaned_train = cleaned_train[selected_features]
unPCA_feature_vecs = []
for i in range(len(cleaned_train)):
    vec = list(cleaned_train.iloc[i])
    unPCA_feature_vecs.append(vec)

"""
    Normalize unPCA'd vectors and apply PCA to them, finding the
    3 most influential variables
"""
for i in range(len(unPCA_feature_vecs)):
    unPCA_feature_vecs[i] = np.array([unPCA_feature_vecs[i]])[0]
pca = PCA(n_components=7)
pca.fit(unPCA_feature_vecs)

"""
    After having visualized the variance ratios, it appears that
    sex, class, and age are major factors in determining
    Titanic survival. Let's plot this and take a closer look:
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(train)):
    vec = train.iloc[i]
    if train.iloc[i]["Survived"] == 0:
        ax.scatter(train.iloc[i]["Sex"], train.iloc[i]["Pclass"], train.iloc[i]["Age"], c="r", marker="+")
    else:
        ax.scatter(train.iloc[i]["Sex"], train.iloc[i]["Pclass"], train.iloc[i]["Age"], c="b", marker="+")
# plt.show()

"""
    Great. The above visualization indicates that men are much more likely
    to die than women. Age has little bearing; however class does a bit.
    Now that we know we need these three factors, let's make a neural network
    and learn it:
"""
x_ = tf.placeholder(tf.float32, shape=[3, None])
y_ = tf.placeholder(tf.float32, shape=[2, None])

w1 = tf.Variable(tf.random_uniform([5, 3], -1, 1))
b1 = tf.Variable(tf.zeros([5, 1]))
w2 = tf.Variable(tf.random_uniform([2, 5], -1, 1))
b2 = tf.Variable(tf.zeros([2, 1]))
# w3 = tf.Variable(tf.random_uniform([2, 5], -1, 1))
# b3 = tf.Variable(tf.zeros([2, 1]))

layer_1 = tf.nn.sigmoid(tf.matmul(w1, x_) + b1)
output = tf.nn.sigmoid(tf.matmul(w2, layer_1) + b2)
# output = tf.nn.sigmoid(tf.matmul(w3, layer_2) + b3)

loss_func = tf.reduce_mean(((y_ * tf.log(output)) + ((1 - y_) * tf.log(1.0 - output))) * -1)
train_step = tf.train.AdamOptimizer(0.01).minimize(loss_func)
n_epochs = 10000

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

"""
    Fetch all rest of training data from csv
"""
feature_vecs = []
outputs = []
train = train.dropna(axis=0, subset=["Sex", "Pclass", "Age"], how="any")

for i in range(len(train)):
    vec = [train.iloc[i]["Sex"], train.iloc[i]["Pclass"], train.iloc[i]["Age"]]
    vec = normalize(np.array([vec]))[0].T
    feature_vecs.append(vec)
    if train.iloc[i]["Survived"] == 0:
        outputs.append(np.array([[1], [0]]))
    else:
        outputs.append(np.array([[0], [1]]))

for i in range(len(feature_vecs)):
    feature_vecs[i] = np.array([feature_vecs[i]]).T

feature_vecs, outputs = shuffle(feature_vecs, outputs)

"""
    Great. Now, we train
"""
for i in range(n_epochs):
    cum_loss = 0
    amt_docs = 0
    for j in range(len(feature_vecs)):
        amt_docs += 1
        sess.run(train_step, feed_dict={x_: feature_vecs[j], y_: outputs[j]})
        loss = sess.run(loss_func, feed_dict={x_: feature_vecs[j], y_: outputs[j]})
        cum_loss += loss
        avg_loss = cum_loss / amt_docs
        if amt_docs % 500 == 0:
            print(avg_loss)






