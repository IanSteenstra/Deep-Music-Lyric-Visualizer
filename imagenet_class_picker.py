from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import pickle


class ImageNetClassPicker:
    def __init__(self):
        self.image_classes = {}
        self.lyrics = []
        self.sent_trans_model = SentenceTransformer('stsb-roberta-large')
        self.sentiment_analyser = SentimentIntensityAnalyzer()

    def load_imagenet_class_labels(self, filename):
        with open(filename) as file:
            for line in file:
                num = line.rstrip().split(':', 1)[0]
                classes = line.rstrip().split(
                    ':', 1)[-1].replace(',', '').replace("'", '')
                self.image_classes[num] = classes

    def load_song_lyrics(self, filename):
        with open(filename) as file:
            for line in file:
                self.lyrics.append(line.rstrip().replace(
                    ',', '').replace("'", ''))

    def save_list(self, filename, list_to_save):
        with open(filename, "wb") as file:
            pickle.dump(list_to_save, file)

    def load_list(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    def get_classes_from_semantic_similarity(self, max_classes_per_lyric=1, max_classes=12):
        class_indexes = set()

        print("Getting {} Classes From Semantic Similarity\n".format(max_classes))

        for lyric in tqdm(self.lyrics):
            temp_key = 0
            temp_cos_sim = -1
            for key in self.image_classes:
                embedding1 = self.sent_trans_model.encode(
                    self.image_classes[key], convert_to_tensor=True)
                embedding2 = self.sent_trans_model.encode(
                    lyric, convert_to_tensor=True)
                if (util.pytorch_cos_sim(embedding1, embedding2).item() > temp_cos_sim):
                    temp_key = key
                    temp_cos_sim = util.pytorch_cos_sim(embedding1, embedding2).item()

            class_indexes.add(temp_key)

        if (len(class_indexes) > max_classes):
            return random.sample(class_indexes, max_classes)
        
        return class_indexes

    def get_classes_from_sentiment_analysis(self):
        lyric_sentiment = self.sentiment_anallysis_helper(
            ' '.join(self.lyrics))

        common_sentiment_list = []
        for key in self.image_classes:
            class_sentiment = self.sentiment_anallysis_helper(
                self.image_classes[key])

            if (class_sentiment == lyric_sentiment):
                common_sentiment_list.append(int(key))

        return common_sentiment_list

    def sentiment_anallysis_helper(self, text):
        sentiment_score = self.sentiment_analyser.polarity_scores(text)

        if sentiment_score['compound'] >= 0:
            sentiment = 'Positive'
        elif sentiment_score['compound'] < 0:
            sentiment = 'Negative'

        return sentiment
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import pickle


class ImageNetClassPicker:
    def __init__(self):
        self.image_classes = {}
        self.lyrics = []
        self.sent_trans_model = SentenceTransformer('stsb-roberta-large')
        self.sentiment_analyser = SentimentIntensityAnalyzer()

    def load_imagenet_class_labels(self, filename):
        with open(filename) as file:
            for line in file:
                num = line.rstrip().split(':', 1)[0]
                classes = line.rstrip().split(
                    ':', 1)[-1].replace(',', '').replace("'", '')
                self.image_classes[num] = classes

    def load_song_lyrics(self, filename):
        with open(filename) as file:
            for line in file:
                lyric_line = line.rstrip().replace(
                    ',', '').replace("'", '')
                if (lyric_line not in self.lyrics):
                    self.lyrics.append(lyric_line)

    def save_list(self, filename, list_to_save):
        with open(filename, "wb") as file:
            pickle.dump(list_to_save, file)

    def load_list(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    def get_classes_from_semantic_similarity(self, max_classes_per_lyric=1, max_classes=12):
        class_indexes = set()

        print("Getting {} Classes From Semantic Similarity\n".format(max_classes))

        for lyric in tqdm(self.lyrics):
            temp_key = 0
            temp_cos_sim = -1
            for key in self.image_classes:
                embedding1 = self.sent_trans_model.encode(
                    self.image_classes[key], convert_to_tensor=True)
                embedding2 = self.sent_trans_model.encode(
                    lyric, convert_to_tensor=True)
                if (util.pytorch_cos_sim(embedding1, embedding2).item() > temp_cos_sim):
                    temp_key = key
                    temp_cos_sim = util.pytorch_cos_sim(embedding1, embedding2).item()

            class_indexes.add(int(temp_key))

        if (len(class_indexes) > max_classes):
            return random.sample(class_indexes, max_classes)
        
        return class_indexes

    def get_classes_from_sentiment_analysis(self):
        lyric_sentiment = self.sentiment_anallysis_helper(
            ' '.join(self.lyrics))

        common_sentiment_list = []
        for key in self.image_classes:
            class_sentiment = self.sentiment_anallysis_helper(
                self.image_classes[key])

            if (class_sentiment == lyric_sentiment):
                common_sentiment_list.append(int(key))

        return common_sentiment_list

    def sentiment_anallysis_helper(self, text):
        sentiment_score = self.sentiment_analyser.polarity_scores(text)

        if sentiment_score['compound'] >= 0:
            sentiment = 'Positive'
        elif sentiment_score['compound'] < 0:
            sentiment = 'Negative'

        return sentiment
