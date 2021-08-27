from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from tqdm import tqdm
from time import sleep
import pickle

class ImageNetClassPicker: 
    def __init__(self):
        self.image_classes = {}
        self.lyrics = []
        self.sent_trans_model = SentenceTransformer('stsb-roberta-large')
        self.sentiment_analyser = SentimentIntensityAnalyzer()

    def load_imagenet_class_labels(self, filename):
        with open(filename) as file:
            while (line := file.readline().rstrip()):
                num = line.split(':', 1)[0]
                classes = line.split(':', 1)[-1].replace(',', '').replace("'", '')
                self.image_classes[num] = classes
    
    def load_song_lyrics(self, filename):
        with open(filename) as file:
            while (line := file.readline().rstrip()):
                self.lyrics.append(line.replace(',', '').replace("'", ''))

    def save_list(self, filename, list_to_save):
        with open(filename, "wb") as file:
            pickle.dump(list_to_save, file)

    def load_list(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    def get_classes_from_semantic_similarity(self, max_classes_per_lyric):
        class_indexes = []

        print("Getting %s Classes From Semantic Similarity" % len(self.lyrics) * max_classes_per_lyric)

        for lyric in tqdm(self.lyrics):
            cosine_scores_list = []

            for key in self.image_classes:
                embedding1 = self.sent_trans_model.encode(self.image_classes[key], convert_to_tensor=True)
                embedding2 = self.sent_trans_model.encode(lyric, convert_to_tensor=True)
                cosine_scores_list.append((key, util.pytorch_cos_sim(embedding1, embedding2).item()))

            temp_cosine_scores_list_sorted = sorted(cosine_scores_list, key=lambda x: float(x[1]), reverse=True)
            class_indexes += [x[0] for x in temp_cosine_scores_list_sorted[:max_classes_per_lyric]]

        return class_indexes

    def get_classes_from_sentiment_analysis(self, max_classes):
        sentiment_score = self.sentiment_analyser.polarity_scores(self.load_song_lyrics)

        if sentiment_score['compound'] >= 0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Positive'
        elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Neutral'
        elif sentiment_score['compound'] <= -0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Negative'

        sentiment
        abs(sentiment_percentage) * 100

        return []