import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')


def load_data(database_filepath):

    """
    Load and merge datasets
    input:
         database name
    outputs:
        X: messages 
        y: category names
        
    """

    engine = create_engine('sqlite:///' + database_filepath)
    query = 'SELECT * FROM sql_dataset'
    df = pd.read_sql(query, engine)
    rows_with_twos = df.isin([2]).any(axis=1)
    filtered_df = df[~rows_with_twos]
    
    X = filtered_df['message']
    y = filtered_df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):

    """
    Tokenizes message data
    
    INPUT:
       text (string): message text data
    
    OUTPUT:
        (DataFrame) clean_messages: array of tokenized message data
    """
        
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)     # Remove non-alphanumeric characters using regex
    words = word_tokenize(text)                    # Tokenize the text into words
    words = [word.lower() for word in words]       # Convert words to lowercase
    stop_words = set(stopwords.words('english'))   # Remove stopwords
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()                      # Initialize Porter Stemmer
    words = [stemmer.stem(word) for word in words] # Apply stemming
    
    return words


def build_model():

    """
    Pipeline construction
      
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [5],    
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model



def evaluate_model(model, X_test, Y_test, category_names):

    """
    Evaluates models performance in predicting message categories
    
    INPUT:
        model (Classification): stored classification model
        X_test (string): Independent Variables
        Y_test (string): Dependent Variables
        category_names (DataFrame): Stores message category labels
        
    OUTPUT:
        Prints a classification report for every category
        
    """

    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred, target_names=category_names)
    print(class_report)
   


def save_model(model, model_filepath):
   
    """
    Saves trained classification model to pickle file
    
    INPUT:
        model (Classification): stored classification model
        model_filepath (string): Filepath to pickle file
    
    """
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()