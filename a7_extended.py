import math, os, pickle, re
from typing import Tuple, List, Dict


class BayesClassifier:
    """A simple BayesClassifier implementation

    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        pos_filename - name of positive dictionary cache file
        neg_filename - name of positive dictionary cache file
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
    """

    def __init__(self):
        """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
        cache of a trained classifier is stored in the current folder it is loaded,
        otherwise the system will proceed through training.  Once constructed the
        classifier is ready to classify input text."""
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.training_data_directory: str = "movie_reviews/"
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"

        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
        else:
            print("Data files not found - running training...")
            self.train()

    def train(self) -> None:
        """Trains the Naive Bayes Sentiment Classifier

        Train here means generates `pos_freq/neg_freq` dictionaries with frequencies of
        words in corresponding positive/negative reviews
        """
        # get the list of file names from the training data directory
        # os.walk returns a generator (feel free to Google "python generators" if you're
        # curious to learn more, next gets the first value from this generator or the
        # provided default `(None, None, [])` if the generator has no values)
        _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
        if not files:
            raise RuntimeError(f"Couldn't find path {self.training_data_directory}")

        for index, filename in enumerate(files, 1):
            print(f"Training on file {index} of {len(files)}")
            text = self.load_file(os.path.join(self.training_data_directory, filename))
            tokens: List[str] = self.tokenize(text)

            if filename.startswith(self.pos_file_prefix):
                self.update_dict(tokens, self.pos_freqs)
            elif filename.startswith(self.neg_file_prefix):
                self.update_dict(tokens, self.neg_freqs)
            else:
                print("Neutral Review Exception")

        self.save_dict(self.pos_freqs, self.pos_filename)
        self.save_dict(self.neg_freqs, self.neg_filename)

    def classify(self, text: str) -> str:
        """Classifies given text as positive, negative or neutral from calculating the
        most likely document class to which the target string belongs

        Args:
            text - text to classify

        Returns:
            classification, either positive, negative or neutral
        """

        tokens: List[str] = self.tokenize(text)

        positive_probability: float = 0
        negative_probability: float = 0
        
        pos_sum = sum(self.pos_freqs.values())
        neg_sum = sum(self.neg_freqs.values())

        for token in tokens:
            pos_occurences = self.pos_freqs.get(token, 0)
            neg_occurences = self.neg_freqs.get(token, 0)

            positive_probability += math.log((pos_occurences + 1) / pos_sum)
            negative_probability += math.log((neg_occurences + 1) / neg_sum)
        
        print("POSITIVE PROBABILITY: ", positive_probability)
        print("NEGATIVE PROBABILITY: ", negative_probability)

        return "positive" if positive_probability > negative_probability else "negative"

    def load_file(self, filepath: str) -> str:
        """Loads text of given file

        Args:
            filepath - relative path to file to load

        Returns:
            text of the given file
        """
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        """Pickles given dictionary to a file with the given name

        Args:
            dict - a dictionary to pickle
            filepath - relative path to file to save
        """
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        """Loads pickled dictionary stored in given file

        Args:
            filepath - relative path to file to load

        Returns:
            dictionary stored in given file
        """
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
        """Splits given text into a list of the individual tokens in order

        Args:
            text - text to tokenize

        Returns:
            tokens of given text in order
        """
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())
        return tokens
    
    def get_stoplist(self):
        with open("sorted_stoplist.txt", "r") as stoplist:
            return stoplist.read()

    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        """Updates given (word -> frequency) dictionary with given words list

        By updating we mean increment the count of each word in words in the dictionary.
        If any word in words is not currently in the dictionary add it with a count of 1.
        (if a word is in words multiple times you'll increment it as many times
        as it appears)

        Args:
            words - list of tokens to update frequencies of
            freqs - dictionary of frequencies to update
        """

        stoplist = self.get_stoplist()
        for word in words:
            if word in stoplist: continue
            if freqs.get(word) is None:
                freqs[word] = 1
            else:
                freqs[word] += 1


if __name__ == "__main__":
    b = BayesClassifier()
    b.get_stoplist()

    pos_denominator = sum(b.pos_freqs.values())
    neg_denominator = sum(b.neg_freqs.values())

    print("\nHere are some sample probabilities.")
    print(f"P('love'| pos) {(b.pos_freqs['love']+1)/pos_denominator}")
    print(f"P('love'| neg) {(b.neg_freqs['love']+1)/neg_denominator}")
    print(f"P('terrible'| pos) {(b.pos_freqs['terrible']+1)/pos_denominator}")
    print(f"P('terrible'| neg) {(b.neg_freqs['terrible']+1)/neg_denominator}")

    # uncomment the below lines once you've implemented `classify`
    print("\nThe following should all be positive.")
    print(b.classify('I love computer science'))
    print(b.classify('this movie is fantastic'))
    print("\nThe following should all be negative.")
    print(b.classify('rainy days are the worst'))
    print(b.classify('computer science is terrible'))