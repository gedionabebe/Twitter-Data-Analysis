from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_processing_and_exploration import DataPreprocessor
from sklearn.metrics import accuracy_score


class SentimentAnalysis:

    def __init__(self):
        self.DataPreprocessor = DataPreprocessor()
    
    def Model_training(self):
        lables, processed_features = self.DataPreprocessor.feature_preparation()
        vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
        processed_features_vectorized = vectorizer.fit_transform(processed_features).toarray()
        X_train, X_test, y_train, y_test = train_test_split(processed_features_vectorized, lables, test_size=0.2, random_state=0)
        randomforest_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        randomforest_classifier.fit(X_train,y_train)
        predictions = randomforest_classifier.predict(X_test)
        score = accuracy_score(y_test,predictions)
        
        return predictions, score

if __name__ == "__main__":
    test = SentimentAnalysis().Model_training()