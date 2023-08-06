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
from sklearn.svm import SVC
import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


def load_data(database_filepath):
    
    """
    Load and merge datasets
    input:
         database name
    outputs:
        X: messages 
        y: category names
        
    """
    engine = create_engine('sqlite:///database_filepath')
    query = 'SELECT * FROM full_dataset'
    df = pd.read_sql(query, engine)
    rows_with_twos = df.isin([2]).any(axis=1)
    filtered_df = df[~rows_with_twos]
    
    X = filtered_df['message']
    y = filtered_df.drop(['id', 'message', 'original', 'genre'], axis = 1)

    return X, y


def tokenize(text):
        
    """
    Tokenizes message data
    
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
   This function build constructs an ML pipeline & 
   splits your data into training and testing &
   and fits the ML pipeline on the training data &
   applies a Gridsearch CV
   
   """
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)
    pipe.fit(X_train, y_train)
    
    from sklearn.model_selection import RandomizedSearchCV
    
    param_grid = {
    'clf__n_estimators': [5]
     }

    cv = GridSearchCV(pipe, param_grid = parameters)
    
    return cv

def evaluate_model(y_test, y_test_pred):
    
    """
    inputs:
        y_test
        y_test predicted
    
    output:
        a classification report scores for every feature
        
    """
    print(classification_report(y_test, y_test_pred))


def save_model(model, model_filename):
    
    """
    Save model to a pickle file
    Input: 
    	your model name
	saved name for your pickle file
	
    """
    model_filename = 'best_model.pkl'
    with open(model_filename, 'wb') as model_file:
    	pickle.dump(model, model_file)
    print(f"Best model saved as {model_filename}")
   

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