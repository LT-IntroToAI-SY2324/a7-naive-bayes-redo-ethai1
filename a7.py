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
        
        # files now holds a list of the filenames
        # self.training_data_directory holds the folder name where these files are
        

        # stored below is how you would load a file with filename given by `fName`
        # `text` here will be the literal text of the file (i.e. what you would see
        # if you opened the file in a text editor


        # *Tip:* training can take a while, to make it more transparent, we can use the
        # enumerate function, which loops over something and has an automatic counter.
        # write something like this to track progress (note the `# type: ignore` comment
        # which tells mypy we know better and it shouldn't complain at us on this line):

        for index, filename in enumerate(files, 1): # type: ignore
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

        # we want to fill pos_freqs and neg_freqs with the correct counts of words from
        # their respective reviews
        
        # for each file, if it is a negative file, update (see the Updating frequencies
        # set of comments for what we mean by update) the frequencies in the negative
        # frequency dictionary. If it is a positive file, update (again see the Updating
        # frequencies set of comments for what we mean by update) the frequencies in the
        # positive frequency dictionary. If it is neither a postive or negative file,
        # ignore it and move to the next file (this is more just to be safe; we won't
        # test your code with neutral reviews)
        

        # Updating frequences: to update the frequencies for each file, you need to get
        # the text of the file, tokenize it, then update the appropriate dictionary for
        # those tokens. We've asked you to write a function `update_dict` that will make
        # your life easier here. Write that function first then pass it your list of
        # tokens from the file and the appropriate dictionary
        

        # for debugging purposes, it might be useful to print out the tokens and their
        # frequencies for both the positive and negative dictionaries
        

        # once you have gone through all the files, save the frequency dictionaries to
        # avoid extra work in the future (using the save_dict method). The objects you
        # are saving are self.pos_freqs and self.neg_freqs and the filepaths to save to
        # are self.pos_filename and self.neg_filename



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
            pos_occurences = self.pos_freqs.get(token, 1)
            neg_occurences = self.neg_freqs.get(token, 1)

            positive_probability += math.log(pos_occurences / pos_sum)
            negative_probability += math.log(neg_occurences / neg_sum)
        
        print("POSITIVE PROBABILITY: ", positive_probability)
        print("NEGATIVE PROBABILITY: ", negative_probability)
        return "positive" if positive_probability > negative_probability else "negative"
    
        # get a list of the individual tokens that occur in text
        

        # create some variables to store the positive and negative probability. since
        # we will be adding logs of probabilities, the initial values for the positive
        # and negative probabilities are set to 0
        

        # get the sum of all of the frequencies of the features in each document class
        # (i.e. how many words occurred in all documents for the given class) - this
        # will be used in calculating the probability of each document class given each
        # individual feature
        

        # for each token in the text, calculate the probability of it occurring in a
        # postive document and in a negative document and add the logs of those to the
        # running sums. when calculating the probabilities, always add 1 to the numerator
        # of each probability for add one smoothing (so that we never have a probability
        # of 0)


        # for debugging purposes, it may help to print the overall positive and negative
        # probabilities
        

        # determine whether positive or negative was more probable (i.e. which one was
        # larger)
        

        # return a string of "positive" or "negative"

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
        
        for word in words:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1


if __name__ == "__main__":
    # uncomment the below lines once you've implemented `train` & `classify`
    b = BayesClassifier()
    # a_list_of_words = ["I", "really", "like", "this", "movie", ".", "I", "hope", \
    #                    "you", "like", "it", "too"]
    # a_dictionary = {}
    # b.update_dict(a_list_of_words, a_dictionary)
    # assert a_dictionary["I"] == 2, "update_dict test 1"
    # assert a_dictionary["like"] == 2, "update_dict test 2"
    # assert a_dictionary["really"] == 1, "update_dict test 3"
    # assert a_dictionary["too"] == 1, "update_dict test 4"
    # print("update_dict tests passed.")

    pos_denominator = sum(b.pos_freqs.values())
    neg_denominator = sum(b.neg_freqs.values())

    # print("\nThese are the sums of values in the positive and negative dicitionaries.")
    # print(f"sum of positive word counts is: {pos_denominator}")
    # print(f"sum of negative word counts is: {neg_denominator}")

    # print("\nHere are some sample word counts in the positive and negative dicitionaries.")
    # print(f"count for the word 'love' in positive dictionary {b.pos_freqs['love']}")
    # print(f"count for the word 'love' in negative dictionary {b.neg_freqs['love']}")
    # print(f"count for the word 'terrible' in positive dictionary {b.pos_freqs['terrible']}")
    # print(f"count for the word 'terrible' in negative dictionary {b.neg_freqs['terrible']}")
    # print(f"count for the word 'computer' in positive dictionary {b.pos_freqs['computer']}")
    # print(f"count for the word 'computer' in negative dictionary {b.neg_freqs['computer']}")
    # print(f"count for the word 'science' in positive dictionary {b.pos_freqs['science']}")
    # print(f"count for the word 'science' in negative dictionary {b.neg_freqs['science']}")
    # print(f"count for the word 'i' in positive dictionary {b.pos_freqs['i']}")
    # print(f"count for the word 'i' in negative dictionary {b.neg_freqs['i']}")
    # print(f"count for the word 'is' in positive dictionary {b.pos_freqs['is']}")
    # print(f"count for the word 'is' in negative dictionary {b.neg_freqs['is']}")
    # print(f"count for the word 'the' in positive dictionary {b.pos_freqs['the']}")
    # print(f"count for the word 'the' in negative dictionary {b.neg_freqs['the']}")

    # print("\nHere are some sample probabilities.")
    # print(f"P('love'| pos) {(b.pos_freqs['love']+1)/pos_denominator}")
    # print(f"P('love'| neg) {(b.neg_freqs['love']+1)/neg_denominator}")
    # print(f"P('terrible'| pos) {(b.pos_freqs['terrible']+1)/pos_denominator}")
    # print(f"P('terrible'| neg) {(b.neg_freqs['terrible']+1)/neg_denominator}")

    # print("\nThe following should all be positive.")
    # print(b.classify('I love computer science'))
    # print(b.classify('this movie is fantastic'))
    # print("\nThe following should all be negative.")
    # print(b.classify('rainy days are the worst'))
    # print(b.classify('computer science is terrible'))

    print("\nThe following is to test out the method with each groups responses")
    # print(b.classify('I am so excited for the solar eclipse! It is going to be so cool!'))
    # print(b.classify('Computer Science is so cool! I hope to take more Computer Science classes in the future!'))
    # print(b.classify('Kung Fu Hustle was hilarious! Definitely one of my favorites.'))
    # print(b.classify('I would put The Empire Strikes Back in my top 5 favorite films'))

    # print(b.classify('The solar eclipse is going to be boring, why should we waste our time?'))
    # print(b.classify('The voice acting was mediocre and the animation was disappointing.'))
    # print(b.classify('Watching this was a huge waste of my precious time'))
    # print(b.classify('I think English is the worst subject.'))

    print(b.classify('Hands down the best thing going for Dune is the stunning visuals from beginning to end. Director Denis Villeneuve compiled a beautiful collection of scenes that are truly a sight to see. One thing you’ll notice is how the cinematography seamlessly compliments so many elements in the film.  Dune is so masterfully shot, that you could watch this movie on mute and still be entertained. Unlike the original film and the TV series,  the plot was easy to follow. Despite the dense, sci-elements in the story, I appreciated the focused approach on only one group/family. It was helpful to explore this fantasy world and all the cool technology through the perspective of just a few characters. As a result, it doesn’t take long to have a sensible grasp of the underlying politics and motivations of various characters. Timothée Chalamet was exceptional as Paul Atreides. His performance made me care about his character’s journey and his potential growth. Oscar Isaac as Duke Leto was also great in his role. He presented such a stoic feel to his character that almost steals every scene he’s featured in. Jason Momoa as Duncan, totally badassery!!! While the role may have been small, this was right in Momoa’s lane especially when it came to his fight scenes. Of all of the roles, Stellan Skarsgård surprised me the most as The Baron. He was completely unrecognizable thanks to both his delivery and the Dune make up team. I am looking forward to the next part!'))
    print(b.classify('This movie has some of the most amazing set pieces I have seen in AAA Sci-Fi films! Definitely a must watch in premium theater formats such as Dolby Cinema & Imax! I recommend Dolby Cinema, the audio was breathtakingly amazing! You can feel the static charges from the worms, and there are scenes where the Dolby Atmos audio extremely intensifies everything, which adds depth to the emotional feel of the movie. The setting and world of Arrakis is visually stunning, and the action gives way to, again, breathtaking periods of time where you are completely visually engulfed in the film. The story itself is not nearly as complicated as I thought it was going to be. It does take a bit to take off in the storyline, but you feel that when the 2nd film comes out everything will fit together nicely. This is definitely part of a continuing story, and the film does not shy away from that. So it does take some patience to see the story trough, but with that in mind one should just sit back and watch this great entry into a very interesting world with well written and acted characters. This film definitely is amazing, a must-see on the big screen and in the best format you can see! Be prepared to leave wanting more, however! There are times where the names and terms can be confusing, just let it ride out everything as it is all explained, and the film easily makes sense of the storyline in time. The acting was supreme, and everything else in between was nothing short of near perfection!'))
    # print(b.classify("Yet another watered version of the already watered down late Star Wars saga, now even more shamefully targeted to the teen public, coated with great scenes, great performances and top celebs. The protagonist depicts once again the typical frail and introspective teenager whose deeper egoistic dream is to be the center of the universe, full of power, glory and in a reality where adults are basically stupid, nuisances or stereotypical buffoons. For me this type of story just keeps cashing in over this teen's distortion of reality, furthering the damage that media has been done to society over the last half century. By making young adults keep thinking they are inherently special and smarter than adults, they will secretly expecting to be discovered. This goes on till they are no longer young. As nothing of special happens in their lives, they feel betrayed, turn into bitter and depressed adults, and so easy targets of other industries such as tobacco, alcohol, processed food, drugs, entertainment, eventually becoming actual stereotypical buffoons for the next generation."))
    # print(b.classify("This movie is being heralded for its stunning cinematography & I can’t argue against that. It is indeed a beautiful composition of shots that pays homage to David Lean’s Lawrence of Arabia. However, story wise it falls flat for me. The narrative & character focus is a bit pretentious with little to deliver and so you’re left with actors holding stoic facial gestures for long periods of time with soundtrack to suit ghe mood. I don’t know who takes the cake for this… Sergio Leone’s Once Upon in the West or this new rendition of Dune. The pandemic has altered our perception of good cinema when a film like this scores 8.5/10 on Rotten Tomatoes. I just don’t get it. The ensemble cast was a necessary feature of this film to keep audiences invested pre & during its theatrical release. If you have 2.5 hours to spare, it’s worth watching but don’t expect to be blown away by the whole story."))