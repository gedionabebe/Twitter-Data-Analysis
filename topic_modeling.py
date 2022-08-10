import warnings
warnings.filterwarnings('ignore')
import gensim
from pprint import pprint
from data_processing_and_exploration import DataPreprocessor



class LDATopicModeling:

    def __init__(self):
        self.preprocess_data = DataPreprocessor()
    
    def TopicModeler(self):
        word_to_id,corpus = self.preprocess_data.preprocess_data()
        lda_model = gensim.models.ldamodel.LdaModel(corpus,
                                           id2word=word_to_id,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        pprint(lda_model.print_topics())
    
if __name__ == "__main__":
    test = LDATopicModeling()
    #test.TopicModeler()