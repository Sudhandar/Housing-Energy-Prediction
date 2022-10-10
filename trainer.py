from sklearn import metrics
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


class Model:
  def __init__(self,regressor, X_train, y_train, X_test, y_test, return_metrics = False, save_predictions = False):
    self.regressor = regressor
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.return_metrics = metrics
    self.save_predictions = save_predictions
  
  def calculate_results(self, y_true, y_pred):

    r2 = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    print('R^2:',r2)
    print('MAE:',mae)
    return r2, mae

  def train_model(self):

    results = {"train":{"r2":0, "mae":0},
               "test":{"r2":0, "mae":0}}
    print("Training Model")
    self.regressor.fit(self.X_train, self.y_train)
    y_train_pred = self.regressor.predict(self.X_train)
    print("Training Results:")
    results["train"]["r2"], results["train"]["mae"] = self.calculate_results(self.y_train, y_train_pred)
    print("Saving Trained model as a pickle file")
    pickle.dump(self.regressor, open('model.pkl','wb'))
    self.regressor = pickle.load(open('model.pkl','rb'))
    print("Making Predictions")
    self.X_test.to_csv("Test Data.csv",index = False)
    y_pred = self.regressor.predict(self.X_test)
    print("Test Results:")
    results["test"]["r2"], results["test"]["mae"] = self.calculate_results(self.y_test, y_pred)
    if self.save_predictions:
      self.store_predictions(y_pred)

    if self.return_metrics:
      return results

  def plot_visualizer(self):
    visualizer = ResidualsPlot(self.regressor, hist = False, qqplot = True)
    visualizer.fit(self.X_train, self.y_train)
    visualizer.score(self.X_test, self.y_test)
    visualizer.show()

  def store_predictions(self, y_pred):
    y_pred_df = pd.DataFrame(y_pred, columns = ['kwh_predicted'], index = X_test.index)
    self.X_test = pd.DataFrame(self.X_test)
    predictions = pd.concat([self.X_test,y_pred_df],axis = 1)
    predictions.to_csv('./predictions/predicted_results.csv',index = False)


def get_training_data(data, target_column, feature_list, split_percentage = 0.2):

  y = data[target_column]
  X = data[feature_list]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= split_percentage , random_state=8)
  return X_train, X_test, y_train, y_test
