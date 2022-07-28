from create_sqlite_db import CreateDatabase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.linear_model import Lasso,LassoCV
from yellowbrick.regressor import ResidualsPlot
from data_utils import CleanDataset
from trainer import *


if __name__ == '__main__':
	creator = CreateDatabase()
	creator.create_tables()
	cleaner = CleanDataset(transform_target_variable = False, log_numerical_features = False, do_label_encoding = True)
	clean_data = cleaner.clean()
	final_features = ['regionc','division','reportable_domain','hdd65','hdd30yr','cdd30yr','dollarel','dolelsph','metromicro','ur','totrooms','heatroom','acrooms','totsqft']
	X_train, X_test, y_train, y_test  = get_training_data(clean_data, 'kwh', final_features, split_percentage = 0.2)
	xgb = XGBRegressor(learning_rate = 0.1, max_depth = 5, min_samples_leaf = 3, n_estimator =  500)
	xgb_model = Model(xgb, X_train, y_train, X_test, y_test, return_metrics = False)
	xgb_model.train_model()

