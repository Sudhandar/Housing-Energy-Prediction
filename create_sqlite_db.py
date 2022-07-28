from pathlib import Path
import sqlite3
from sqlite3 import Error
from data_utils import Preprocessing


class CreateDatabase:
	def __init__(self):
		print("Creating Database")
		self.db_file = 'recs_2009_survey.db'
		Path(self.db_file).touch()

	def create_connection(self):
	    """ create a database connection to a SQLite database """
	    conn = None
	    try:
	        conn = sqlite3.connect(self.db_file)
	        return conn
	    except Error as e:
	        print(e)

	def create_tables(self):

		preprocessor = Preprocessing()
		data, variable_enum, column_tuple = preprocessor.preprocess_data()
		variable_column_names = tuple(list(variable_enum.columns))
		conn = self.create_connection()
		c = conn.cursor()
		print("Creating Tables in Database")
		survey_table_query = f"CREATE TABLE survey_data {column_tuple}".replace("'",'')
		c.execute(survey_table_query)
		data.to_sql('survey_data', conn, if_exists='append', index = False)
		c.execute(f"CREATE TABLE variable_information {variable_column_names}")
		variable_enum.to_sql('variable_information', conn, if_exists='append', index = False)
		print("Database Created")