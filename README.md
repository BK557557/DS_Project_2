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


