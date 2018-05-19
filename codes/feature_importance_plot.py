import numpy as np
import mungetools as mg
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt  


file_data = open("data/credit-g.txt").readlines()


features_list = []


for line in file_data:
    if '@' in line and '{' in line:
        feature = line.split()[1]
        features_list.append(feature)

# print features_list

features_list = np.asarray(features_list)

print type(features_list)

input_df=mg.loadData()
X = input_df.values[:, 1:]
y = input_df.values[:, 0]
survived_weight = .75
y_weights = np.array([survived_weight if s == 0 else 1 for s in y])
 
print "Rough fitting a RandomForest to determine feature importance..."
forest = RandomForestClassifier(oob_score=True, n_estimators=10)
forest.fit(X, y, sample_weight=y_weights)
feature_importance = forest.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

print feature_importance

fi_threshold = 30   
important_idx = np.where(feature_importance > fi_threshold)[0]

print 'important_idx:'+str(important_idx)+str(type(important_idx))

important_features = features_list[important_idx]
print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance)...\n"
#important_features
sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
#get the figure about important features
pos = np.arange(sorted_idx.shape[0]) + .5


plt.subplot(1, 2, 2)
plt.title('Feature Importance')
plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], color='r',align='center')
plt.yticks(pos, important_features[sorted_idx[::-1]])
plt.xlabel('Relative Importance')
plt.draw()
plt.show()