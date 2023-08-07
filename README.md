# Disaster Response Pipeline Project

## Installation

This repository was written in HTML and Python , and requires the following Python packages: pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys, warnings.

## Project Overview

This code is designed to iniate a web app which an emergency operators could exploit during a disaster, to classify a disaster text messages into several categories which then can be transmited to the responsible entity

The app built to have an ML model to categorize every message received

## File Description:
process_data.py: This python executable code takes as its input CSV files containing message data and message categories (labels), and then creates a SQL database

train_classifier.py: This code trains the ML model with the SQL data base

ETL Pipeline Preparation.ipynb: process_data.py is the development procces for the ETL pipeline

ML Pipeline Preparation.ipynb: train_classifier.py is the development procces for the ML pipeline

data: This folder contains sample messages and categories datasets in csv format.

app: cointains the run.py to iniate the web app.

## Instructions:

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database - 

```python data/process_data.py data/messages.csv data/categories.csv data/disaster_response_pipeline.db```

To run ML pipeline that trains classifier and saves - 

```python models/train_classifier_updated.py data/disaster_response_pipeline.db models/classifier.pkl```

Run the following command in the app's directory to run your web app - 

```python run.py```

![Screen Shot 2023-08-08 at 12 01 55 AM](https://github.com/BK557557/DS_Project_2/assets/141200544/0f5aafc6-55bf-4768-b20e-38b3f6b7db6e)



## Licensing, Authors, Acknowledgements

Many thanks to Figure-8 for making this available to Udacity for training purposes. Special thanks to udacity for the training. Feel free to utilize the contents of this while citing me, udacity, and/or figure-8 accordingly.

NOTICE: Preparation folder is not necessary for this project to run.

