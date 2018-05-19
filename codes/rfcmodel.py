import sys
import numpy as np
import pandas as pd
import cmd_parser as cmd
from sklearn.ensemble import RandomForestClassifier as rfc



if __name__ == '__main__':
	num = 4

	train_data_file,test_data_file,test_result_dir = cmd.TrainTestFileParser(sys.argv,num)
	# test_result_file = open(test_result_dir+"rfc_test_result"+str(num)+".txt",'w+')


	trdata=pd.read_csv(train_data_file,header=None,sep=' ')
	tedata=pd.read_csv(test_data_file,header=None,sep=' ')


	#depthlist = [5,10,15,20,50,100]
	model= rfc(n_estimators=5000,oob_score=True,max_features = None, max_depth = 10)
	model = model.fit(trdata.iloc[:,1:],trdata.iloc[:,0])
	accur = model.score(tedata.iloc[:,1:],tedata.iloc[:,0])
	# resultClass = model.predict(tedata.iloc[:,1:])
	# #resultLogProba = model.predict_log_proba(tedata.iloc[:,1:])
	# resultProba = model.predict_proba(tedata.iloc[:,1:])


	# for x, y in zip(resultClass, resultProba):
	# 	test_result_file.write(str(x)+' ')
	# for z in y:
	# 	test_result_file.write(str(z)+' ')
	# 	test_result_file.write('\n')


	# print len(resultProba)
	# print('Test data accuracy: %f\n' %accur)
	print('Out of Bag accuracy: %f \n' %model.oob_score_)


	exit()