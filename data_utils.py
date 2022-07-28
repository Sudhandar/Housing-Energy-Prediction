
import pandas as pd
import sqlite3
from sqlite3 import Error
from sklearn.preprocessing import LabelEncoder
import numpy as np


def create_connection(db_file = 'recs_2009_survey.db'):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

class CleanDataset:

  def __init__(self, transform_target_variable, log_numerical_features, do_label_encoding):
    self.transform_target_variable = transform_target_variable
    self.log_numerical_features = log_numerical_features
    self.do_label_encoding = do_label_encoding
    self.conn = create_connection()
    self.survey_data = pd.read_sql("SELECT * FROM survey_data", self.conn)

  def get_columns_with_negative_values(self):

    variable_information = pd.read_sql("SELECT * FROM variable_information", self.conn)
    numerical = variable_information[variable_information['variable_type']=='numerical']
    numerical_na = numerical[(numerical['response_labels']=='Not Applicable')|(numerical['response_labels']=='Refuse')|(numerical['response_labels']=="Don't know")]
    numerical_na = numerical_na[numerical_na['variable_name']!='doeid']
    print('Number of numerical columns with negative values:',numerical_na.shape[0])
    numerical_na_columns = list(set(numerical_na['variable_name'].values.tolist()))
    categorical = variable_information[variable_information['variable_type']=='categorical']
    categorical_na = categorical[(categorical['response_labels']=='Not Applicable')|(categorical['response_labels']=='Refuse')|(categorical['response_labels']=="Don't know")]
    print('Number of categorical columns with negative values:',categorical_na.shape[0])
    categorical_na_columns = list(set(categorical_na['variable_name'].values.tolist()))

    return numerical_na_columns, categorical_na_columns

  def handle_numerical_negative_values(self, numerical_na_columns):

    for column in numerical_na_columns:
      self.survey_data.loc[self.survey_data[column]<0, column] = np.nan
    
    total = self.survey_data[numerical_na_columns].isnull().sum().sort_values(ascending=False)
    percent = (self.survey_data[numerical_na_columns].isnull().sum()/self.survey_data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])
    missing_data.reset_index(inplace = True)
    missing_data.rename(columns = {'index':'variable_name'}, inplace = True)
    missing_data.dropna(inplace = True)
    remove_columns = missing_data[missing_data['percent']>0.5]
    remove_columns = remove_columns['variable_name'].values.tolist()
    self.survey_data.drop(remove_columns, axis = 1, inplace = True)
    print("No. of numerical columns removed:", len(remove_columns))
    numerical_columns = missing_data[missing_data['percent']<0.5]
    numerical_columns = numerical_columns['variable_name'].values.tolist()
    with_ac = self.survey_data[(self.survey_data['aircond']=='1')&(self.survey_data['acrooms']!=-2)]
    ac_median = with_ac['acrooms'].median()
    ac_median = int(ac_median)
    self.survey_data.loc[self.survey_data['aircond']=='0', 'acrooms'] = 0
    self.survey_data.loc[self.survey_data['acrooms'].isnull(), 'acrooms'] = ac_median
    numerical_columns.remove('acrooms')
    median_value = self.survey_data.filter(numerical_columns).median()
    self.survey_data[numerical_columns]=self.survey_data[numerical_columns].fillna(median_value.iloc[0])

  def handle_categorical_negative_values(self, categorical_na_columns):

    for column in categorical_na_columns:
      self.survey_data.loc[self.survey_data[column]=='-2', column] = np.nan
      self.survey_data.loc[self.survey_data[column]=='-8', column] = np.nan
      self.survey_data.loc[self.survey_data[column]=='-9', column] = np.nan
      self.survey_data.loc[self.survey_data[column]==-2, column] = np.nan
      self.survey_data.loc[self.survey_data[column]==-8, column] = np.nan
      self.survey_data.loc[self.survey_data[column]==-9, column] = np.nan

    total = self.survey_data[categorical_na_columns].isnull().sum().sort_values(ascending=False)
    percent = (self.survey_data[categorical_na_columns].isnull().sum()/self.survey_data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])
    missing_data.reset_index(inplace = True)
    missing_data.rename(columns = {'index':'variable_name'}, inplace = True)
    missing_data.dropna(inplace = True)
    remove_columns = missing_data[missing_data['percent']>0.5]
    remove_columns = remove_columns['variable_name'].values.tolist()
    self.survey_data.drop(remove_columns, axis = 1, inplace = True)
    print("No of categorical columns removed:", len(remove_columns))
    categorical_columns = missing_data[missing_data['percent']<0.5]
    categorical_columns = categorical_columns['variable_name'].values.tolist()
    mode_value = self.survey_data.filter(categorical_columns).mode()
    self.survey_data[categorical_columns]=self.survey_data[categorical_columns].fillna(mode_value.iloc[0])


  def remove_imputation_flags(self):

    imputation_flags = [column for column in self.survey_data.columns if column[0]=='z']
    self.survey_data.drop(imputation_flags, axis = 1, inplace = True)

  def perform_box_cox_transform(self, column):

    self.survey_data[column], _ = boxcox(self.survey_data[column])


  def perform_log_transform_features(self):

    numerical_columns, _ = self.get_current_columns_list()
    for column in numerical_columns:
      self.survey_data = self.survey_data[self.survey_data[column]>=0]
      self.survey_data[column] = self.survey_data[column].apply(lambda x:np.log(x+1))

  def subset_by_iqr(self, df, column, whisker_width=1.5):

      q1 = df[column].quantile(0.25)                 
      q3 = df[column].quantile(0.75)
      iqr = q3 - q1
      filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
      return df.loc[filter]
    
  def remove_outliers(self):

    numerical_columns, _ = self.get_current_columns_list()
    for column in numerical_columns:
      self.survey_data = self.survey_data[self.survey_data[column]>=0]

    if self.transform_target_variable:
      mean, std = self.survey_data["kwh"].mean(), self.survey_data["kwh"].std()
      cut_off = 3*std
      lower, upper = mean - cut_off, mean + cut_off
      outliers = self.survey_data[(self.survey_data['kwh'] > upper) | (self.survey_data['kwh'] < lower)]
      self.survey_data = self.survey_data[(self.survey_data['kwh'] <= upper) | (self.survey_data['kwh'] >= lower)]
    else:
      self.survey_data = self.subset_by_iqr(self.survey_data, "kwh", whisker_width=1.5)

  def overfit_reducer(self):
      """
      This function takes in a dataframe and returns a list of features that are overfitted.
      """
      overfit = []
      for column in self.survey_data.columns:
          counts = self.survey_data[column].value_counts()
          zeros = counts.iloc[0]
          if zeros / len(self.survey_data) * 100 > 90:
              overfit.append(column)
      overfit = list(overfit)
      self.survey_data.drop(overfit, axis = 1, inplace = True)


  def get_current_columns_list(self):

      target_variable = 'kwh'
      id_variable = "doeid"
      processed_column_list = list(self.survey_data.columns)
      processed_column_list.remove(target_variable)
      processed_column_list.remove(id_variable)
      processed_column_tuple = tuple(processed_column_list)
      numerical_variables = pd.read_sql(f"SELECT DISTINCT variable_name FROM variable_information WHERE variable_type = 'numerical' AND variable_name IN {processed_column_tuple}", self.conn)
      numerical_variables = list(set(numerical_variables['variable_name'].values.tolist()))
      categorical_variables = pd.read_sql(f"SELECT DISTINCT variable_name FROM variable_information WHERE variable_type = 'categorical' AND variable_name IN {processed_column_tuple}", self.conn)
      categorical_variables = list(set(categorical_variables['variable_name'].values.tolist()))

      return numerical_variables, categorical_variables

  def perform_label_encoding(self):
      _, categorical_variables = self.get_current_columns_list()
      labelencoder = LabelEncoder()
      for column in categorical_variables:
        self.survey_data[column] = labelencoder.fit_transform(self.survey_data[column])


  def cast_string_as_categorical(self):

      _, categorical_variables = self.get_current_columns_list()
      for column in categorical_variables:
        self.survey_data[categorical_variables] = self.survey_data[categorical_variables].astype('category')

  def remove_btu_columns(self):
      btu_columns = [column for column in self.survey_data.columns if "btu" in column]
      self.survey_data.drop(btu_columns,axis = 1,  inplace = True)

  def clean(self):
      numerical_na_columns, categorical_na_columns = self.get_columns_with_negative_values()
      self.handle_numerical_negative_values(numerical_na_columns)
      self.handle_categorical_negative_values(categorical_na_columns)
      self.remove_imputation_flags()
      self.remove_btu_columns()
      if self.transform_target_variable:
        self.perform_box_cox_transform("kwh")
      if self.log_numerical_features:
        self.perform_log_transform_features()
      self.remove_outliers()
      self.overfit_reducer()
      self.cast_string_as_categorical()
      if self.do_label_encoding:
        self.perform_label_encoding()
      print("Final DataFrame shape after cleaning:", self.survey_data.shape)
      return self.survey_data

class Preprocessing:
  def __init__(self):
    print("Preprocessing Data")
    self.data = pd.read_csv('./data/recs2009_public.csv')
    self.variable_enum = pd.read_excel('./data/recs2009_public_codebook.xlsx', sheet_name='Codebook', header = None)

  def set_column_snake_case(self):
    self.data.columns = [column.lower().replace(' ','_').strip() for column in self.data.columns]
    self.variable_enum.columns = [column.lower().replace(' ','_').strip() for column in self.variable_enum.columns]
    self.processed_variable_enum = None

  def get_column_data_types_sql(self, numerical_columns, categorical_columns):

    column_types = {}
    for column in numerical_columns:
      if column == 'kwoth' or column =='numcords':
        self.data[column] = self.data[column].apply(float)
        column_types[column] = 'real'
      elif column == 'nkrgalnc' or column == 'nocrcash':
        self.data.loc[self.data[column]=='.',column] = -2
        self.data[column] = self.data[column].apply(int)
        column_types[column] = 'int'
      else:
        self.data[column] = self.data[column].apply(int)
        if column == 'doeid':
          column_types[column] = 'int' + ' '+ 'PRIMARY KEY'
        else:
          column_types[column] = 'int'
    for column in categorical_columns:
      column_types[column] = 'varchar'
    column_tuple = tuple([column + ' ' + column_types[column] for column in self.data.columns])
    return column_tuple

  def split_and_expand(self,df, expand_column_name, index_column_names):
    temp_data = pd.DataFrame(df[expand_column_name].str.split('\n').values.tolist(), index = df.index)
    temp_data = pd.merge(df[index_column_names], temp_data, left_index = True, right_index = True, how = 'inner')
    temp_data = temp_data.melt(id_vars = index_column_names)
    temp_data.dropna(inplace=True)
    temp_data.rename(columns = {'value':expand_column_name}, inplace = True)
    temp_data[expand_column_name] = temp_data[expand_column_name].str.strip()
    return temp_data

  def preprocess_data(self):

    self.variable_enum.columns = self.variable_enum.iloc[1]
    self.variable_enum.drop(self.variable_enum.index[1])
    self.variable_enum = self.variable_enum.iloc[3:,:4]
    self.variable_enum.rename(columns = {list(self.variable_enum)[3]:'Response Labels'}, inplace=True)
    self.set_column_snake_case()
    self.variable_enum['response_codes_and_labels'] = self.variable_enum['response_codes_and_labels'].str.strip()
    variable_enum_no_codes = self.variable_enum[self.variable_enum['response_codes_and_labels'].isnull()]
    self.variable_enum.dropna(inplace = True)
    code_split = self.split_and_expand(self.variable_enum,'response_codes_and_labels',['variable_name','variable_description'])
    label_split = self.split_and_expand(self.variable_enum,'response_labels',['variable_name','variable_description'])
    processed_variable_enum = pd.merge(code_split,label_split, on = ['variable_name','variable_description','variable'], how='inner' )
    processed_variable_enum.sort_values(by = ['variable_name','variable'], inplace = True)
    processed_variable_enum.drop('variable', axis =1, inplace = True)
    processed_variable_enum.rename(columns = {'response_codes_and_labels':'response_codes'}, inplace = True )
    processed_variable_enum_numerical = processed_variable_enum[processed_variable_enum['response_codes'].str.contains(' - ')]
    processed_variable_enum_numerical = processed_variable_enum_numerical[['variable_name']].drop_duplicates()
    processed_variable_enum_numerical['variable_type'] = 'numerical'
    processed_variable_enum_cat = processed_variable_enum[~processed_variable_enum['variable_name'].isin(processed_variable_enum_numerical['variable_name'])]
    processed_variable_enum_cat = processed_variable_enum_cat[['variable_name']].drop_duplicates()
    processed_variable_enum_cat['variable_type'] = 'categorical'
    processed_variable_enum_types = processed_variable_enum_numerical.append(processed_variable_enum_cat)
    variable_enum_no_codes.rename(columns = {'response_codes_and_labels':'response_codes'}, inplace = True )
    variable_enum_no_codes = variable_enum_no_codes[:-1]
    variable_enum_no_codes.dropna(subset=['variable_name'], inplace = True)
    variable_enum_no_codes['variable_type'] = 'numerical'
    processed_variable_enum = pd.merge(processed_variable_enum, processed_variable_enum_types, on = 'variable_name', how='left')
    processed_variable_enum = processed_variable_enum.append(variable_enum_no_codes)
    processed_variable_enum.dropna(subset = ['variable_name'], inplace = True)
    processed_variable_enum = processed_variable_enum[~processed_variable_enum['variable_name'].str.contains('Note:')]
    processed_variable_enum['variable_name'] = processed_variable_enum['variable_name'].apply(lambda x: x.lower().strip().replace(' ','_'))
    variable_column_names = tuple(processed_variable_enum.columns)
    numerical_columns = processed_variable_enum[processed_variable_enum['variable_type']=='numerical']
    numerical_columns = list(set(numerical_columns['variable_name'].values.tolist()))
    categorical_columns = processed_variable_enum[processed_variable_enum['variable_type']=='categorical']
    categorical_columns = list(set(categorical_columns['variable_name'].values.tolist()))
    column_tuple = self.get_column_data_types_sql(numerical_columns, categorical_columns)
    print("Preprocessing Completed")

    return self.data, processed_variable_enum, column_tuple
