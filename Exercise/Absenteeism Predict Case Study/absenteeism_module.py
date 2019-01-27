#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# In[ ]:


class absenteeism_model():
    
	def __init__(self, model_file, scaler_file):
		# read the 'model' and 'scaler
		with open(model_file,'rb') as model_file, open(scaler_file, 'rb') as scaler_file:
			self.reg = pickle.load(model_file)
			self.scaler = pickle.load(scaler_file)
			self.data = None
    
	# take a data file (*.csv) and preprocess it in the same way as in the lectures
	def load_and_clean_data(self, data_file):
		raw_csv_data = pd.read_csv(data_file)
		dataframe = raw_csv_data.copy()
		pd.options.display.max_columns = None
		pd.options.display.max_rows = None
		dataframe = dataframe.drop('ID', axis=1)
		reason_columns = pd.get_dummies(dataframe['Reason for Absence'])
		reason_columns.drop(0,axis=1)
		dataframe = dataframe.drop('Reason for Absence', axis=1)
		reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
		reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
		reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
		reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
		dataframe = pd.concat([dataframe,reason_type_1,reason_type_2,reason_type_3,reason_type_4], axis=1)
		#now each row has a class for its absence
		#rename the class columns a proper name
		column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
               'Daily Work Load Average', 'Body Mass Index', 'Education',
               'Children', 'Pets',  'Reason 1', 'Reason 2', 'Reason 3', 'Reason 4']
		dataframe.columns = column_names
		#reorder columns
		column_names_reordered = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
               'Daily Work Load Average', 'Body Mass Index', 'Education',
               'Children', 'Pets']
		dataframe = dataframe[column_names_reordered]
		#creating checkpoints.... saving the current dataframe in a backup
		dataframe_reasons_merged = dataframe.copy()
		dataframe_reasons_merged['Date'] = pd.to_datetime(dataframe_reasons_merged['Date'], format = '%d/%m/%Y')
		#need to split date to day month and year. get months of all rows in a list
		months_list = []
		for i in range(dataframe_reasons_merged.shape[0]):
			months_list.append(dataframe_reasons_merged['Date'][i].month)
		#merge
		dataframe_reasons_merged['Months'] = months_list
		#now get day of the week 0 is monday...6 is sunday
		day_of_the_week_list = []
		
		for i in range(dataframe_reasons_merged.shape[0]):
			day_of_the_week_list.append(dataframe_reasons_merged['Date'][i].weekday())
			
		#merge
		dataframe_reasons_merged['Day of Week'] = day_of_the_week_list
		#since now merged remove date column
		dataframe_reasons_merged = dataframe_reasons_merged.drop('Date', axis = 1)
		#reorder columns
		column_names_reordered = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Months', 'Day of Week','Transportation Expense', 'Distance to Work', 'Age',
               'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 'Pets']
		dataframe_reasons_merged = dataframe_reasons_merged[column_names_reordered]
		#another checkpoint
		dataframe_reasons_date_merged = dataframe_reasons_merged.copy()
		dataframe_reasons_date_merged['Education'] = dataframe_reasons_date_merged['Education'].map({1:0, 2:1, 3:1, 4:1})  #so if highschool education, class 0. else class 1
		
		#drop unwanted columns
		dataframe_reasons_date_merged = dataframe_reasons_date_merged.drop(["Distance to Work","Daily Work Load Average", "Day of Week"],axis=1)
		# replace the NaN values
		dataframe_reasons_date_merged = dataframe_reasons_date_merged.fillna(value=0)
		self.preprocessed_data = dataframe_reasons_date_merged.copy()
		
		unscaled_inputs = dataframe_reasons_date_merged.copy()
		cols_to_scale =['Months','Transportation Expense', 'Age','Body Mass Index','Children', 'Pets']
		dataframe_reasons_date_merged[cols_to_scale] = self.scaler.transform(dataframe_reasons_date_merged[cols_to_scale])
		unscaled_inputs[cols_to_scale] = dataframe_reasons_date_merged[cols_to_scale]
		# we need this line so we can use it in the next functions
		self.data = unscaled_inputs
		
	# a function which outputs the probability of a data point to be 1
	def predicted_probability(self):
		if (self.data is not None):
			pred = self.reg.predict_proba(self.data)[:,1]
			return pred

	# a function which outputs 0 or 1 based on our model
	def predicted_output_category(self):
		if (self.data is not None):
			pred_outputs = self.reg.predict(self.data)
			return pred_outputs
			
	# predict the outputs and the probabilities and add columns with these values at the end of the new data
	def predicted_outputs(self):
		if (self.data is not None):
			self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
			self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
			return self.preprocessed_data

