import os
import codecs
from tensorflow.python.platform import gfile
from keras.preprocessing import sequence
import numpy as np

class VocabHandler(object):
    """Herramienta para la manipulacion de diccionarios foneticos
    Se asume que el diccionario contiene un par palabra-pronunciacion por linea

    Args:
    train_path: path for .dic file.
    """
    # simbolos especiales para el vocabulario (deben estar presentes siempre).
    _PAD = "_PAD"
    _GO = "_GO"
    _EOS = "_EOS"
    _UNK = "_UNK"
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]

    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    # listas para el entrenamiento, validacion y pruebas
    train_ph_ids = []
    train_gr_ids = []
    valid_ph_ids = []
    valid_gr_ids = []
    test_ph_ids = []
    test_gr_ids = []

    train_gr_length = 0
    valid_gr_length = 0
    test_gr_length = 0
    train_ph_length = 0
    valid_ph_length = 0
    test_ph_length = 0

    max_input_length = 0
    max_output_length = 0
    
    valid_gr = []
    valid_ph = []
    test_gr = []
    test_ph = []
    gr_vocab = {}
    ph_vocab = {}
    gr_size = 0
    ph_size = 0

    #constructor
    def __init__(self, train_path):
        self.train_path = train_path
        self.prepare_g2p_data()
    ## helpers for encode new words
    def encodeWord(self, word, padded=False, one_hot=False):
        w_ids = [self.symbols_to_ids(list(word.upper()), self.gr_vocab)]
        if padded:
            w_ids = sequence.pad_sequences(w_ids, maxlen=self.max_input_length)
            if one_hot:
                w_ids = np.array([self.onehot(w_ids[0], max_len=self.gr_size)])
        return w_ids
        
    def decodePhoneme(self,phoneme_ids, one_hot=False):
        ph_list = []
        if one_hot:
            phoneme_ids = np.argmax(phoneme_ids, axis=1)    
        for id in phoneme_ids:
            if id > self.UNK_ID:
                ph_list.append(self.ph_vocab.keys()[self.ph_vocab.values().index(id)])
        return " ".join(ph_list)
    
    def decodeWord(self,word_ids, one_hot=False):
        gr_list = []
        if one_hot:
            word_ids = np.argmax(word_ids, axis=1)    
        for id in word_ids:
            if id > self.UNK_ID:
                gr_list.append(self.gr_vocab.keys()[self.gr_vocab.values().index(id)])
        return " ".join(gr_list)

    def onehot(self, X, max_len=0):
        X_oh = []
        for x in X:
            X_oh.append([int(x == i) for i in range(max_len)])

        return np.array(X_oh)

    def create_vocabulary(self, data):
        """Create vocabulary from input data.
        Input data is assumed to contain one word per line.

        Args:
        data: word list that will be used to create vocabulary.

        Rerurn:
        vocab: vocabulary dictionary. In this dictionary keys are symbols
                and values are their indexes.
        """
        vocab = {}
        for line in data:
            for item in line:
                if item in vocab:
                    vocab[item] += 1
                else:
                    vocab[item] = 1
        vocab_list = self._START_VOCAB + sorted(vocab)
        vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
        return vocab


    def save_vocabulary(self, vocab, vocabulary_path):
        """Save vocabulary file in vocabulary_path.
        We write vocabulary to vocabulary_path in a one-token-per-line format,
        so that later token in the first line gets id=0, second line gets id=1,
        and so on.

        Args:
        vocab: vocabulary dictionary.
        vocabulary_path: path where the vocabulary will be created.

        """
        print("Creating vocabulary %s" % (vocabulary_path))
        with codecs.open(vocabulary_path, "w", "utf-8") as vocab_file:
            for symbol in sorted(vocab, key=vocab.get):
                vocab_file.write(symbol + '\n')


    def load_vocabulary(self, vocabulary_path, reverse=False):
        """Load vocabulary from file.
        We assume the vocabulary is stored one-item-per-line, so a file:
        d
        c
        will result in a vocabulary {"d": 0, "c": 1}, and this function may
        also return the reversed-vocabulary [0, 1].

        Args:
        vocabulary_path: path to the file containing the vocabulary.
        reverse: flag managing what type of vocabulary to return.

        Returns:
        the vocabulary (a dictionary mapping string to integers), or
        if set reverse to True the reversed vocabulary (a list, which reverses
        the vocabulary mapping).

        Raises:
        ValueError: if the provided vocabulary_path does not exist.
        """
        rev_vocab = []
        with codecs.open(vocabulary_path, "r", "utf-8") as vocab_file:
            rev_vocab.extend(vocab_file.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        if reverse:
            return rev_vocab
        else:
            return dict([(x, y) for (y, x) in enumerate(rev_vocab)])


    def save_params(self, num_layers, size, model_dir):
        """Save model parameters in model_dir directory.

        Returns:
        num_layers: Number of layers in the model;
        size: Size of each model layer.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Save model's architecture
        with open(os.path.join(model_dir, "model.params"), 'w') as param_file:
            param_file.write("num_layers:" + str(num_layers) + "\n")
            param_file.write("size:" + str(size))


    def load_params(self, model_path):
        """Load parameters from 'model.params' file.

        Returns:
        num_layers: Number of layers in the model;
        size: Size of each model layer.
        """
        # Checking model's architecture for decode processes.
        if gfile.Exists(os.path.join(model_path, "model.params")):
            params = open(os.path.join(model_path, "model.params")).readlines()
            for line in params:
                split_line = line.strip().split(":")
                if split_line[0] == "num_layers":
                    num_layers = int(split_line[1])
                if split_line[0] == "size":
                    size = int(split_line[1])
        return num_layers, size


    def symbols_to_ids(self, symbols, vocab):
        """Turn symbols into ids sequence using given vocabulary file.

        Args:
        symbols: input symbols sequence;
        vocab: vocabulary (a dictionary mapping string to integers).

        Returns:
        ids: output sequence of ids.
        """
        ids = [vocab.get(s, self.UNK_ID) for s in symbols]
        return ids


    def split_to_grapheme_phoneme(self, inp_dictionary):
        """Split input dictionary into two separate lists with graphemes and phonemes.

        Args:
        inp_dictionary: input dictionary.
        """
        graphemes, phonemes = [], []
        for line in inp_dictionary:
            split_line = line.strip().split()
            if len(split_line) > 1:
                graphemes.append(list(split_line[0]))
                phonemes.append(split_line[1:])
        return graphemes, phonemes


    def collect_pronunciations(self, dic_lines):
        '''Create dictionary mapping word to its different pronounciations.
        '''
        dic = {}
        for line in dic_lines:
            lst = line.strip().split()
            if len(lst) > 1:
                if lst[0] not in dic:
                    dic[lst[0]] = [" ".join(lst[1:])]
                else:
                    dic[lst[0]].append(" ".join(lst[1:]))
            elif len(lst) == 1:
                print("WARNING: No phonemes for word '%s' line ignored" % (lst[0]))
        return dic


    def split_dictionary(self, train_path, valid_path=None, test_path=None):
        """Split source dictionary to train, validation and test sets.
        """
        source_dic = codecs.open(self.train_path, "r", "utf-8").readlines()
        train_dic, valid_dic, test_dic = [], [], []
        if valid_path:
            valid_dic = codecs.open(valid_path, "r", "utf-8").readlines()
        if test_path:
            test_dic = codecs.open(test_path, "r", "utf-8").readlines()

        dic = self.collect_pronunciations(source_dic)

        # Split dictionary to train, validation and test (if not assigned).
        for i, word in enumerate(dic):
            for pronunciations in dic[word]:
                if i % 20 == 0 and not valid_path:
                    valid_dic.append(word + ' ' + pronunciations)
                elif (i % 20 == 1 or i % 20 == 2) and not test_path:
                    test_dic.append(word + ' ' + pronunciations)
                else:
                    train_dic.append(word + ' ' + pronunciations)
        return train_dic, valid_dic, test_dic


    def prepare_g2p_data(self, model_dir=None, valid_path=None, test_path=None):
        """Create vocabularies into model_dir, create ids data lists.

        Args:
        model_dir: directory in which the data sets will be stored;
        train_path: path to training dictionary;
        valid_path: path to validation dictionary;
        test_path: path to test dictionary.

        Returns:
        A tuple of 6 elements:
            (1) Sequence of ids for Grapheme training data-set,
            (2) Sequence of ids for Phoneme training data-set,
            (3) Sequence of ids for Grapheme development data-set,
            (4) Sequence of ids for Phoneme development data-set,
            (5) Grapheme vocabulary,
            (6) Phoneme vocabulary.
        """
        # Create train, validation and test sets.
        train_dic, valid_dic, test_dic = self.split_dictionary(self.train_path, valid_path, test_path)
        # Split dictionaries into two separate lists with graphemes and phonemes.
        train_gr, train_ph = self.split_to_grapheme_phoneme(train_dic)
        self.valid_gr, self.valid_ph = self.split_to_grapheme_phoneme(valid_dic)
        self.test_gr, self.test_ph = self.split_to_grapheme_phoneme(test_dic)
        # Create vocabularies of the appropriate sizes.
        print("Creating vocabularies in %s" % model_dir)
        self.ph_vocab = self.create_vocabulary(train_ph)
        self.gr_vocab = self.create_vocabulary(train_gr)

        if model_dir:
            self.save_vocabulary(self.ph_vocab, os.path.join(model_dir, "vocab.phoneme"))
            self.save_vocabulary(self.gr_vocab, os.path.join(model_dir, "vocab.grapheme"))

        # Create ids for the training data.
        self.train_ph_ids = [self.symbols_to_ids(line, self.ph_vocab) for line in train_ph]
        self.train_gr_ids = [self.symbols_to_ids(line, self.gr_vocab) for line in train_gr]
        self.valid_ph_ids = [self.symbols_to_ids(line, self.ph_vocab) for line in self.valid_ph]
        self.valid_gr_ids = [self.symbols_to_ids(line, self.gr_vocab) for line in self.valid_gr]
        self.test_ph_ids = [self.symbols_to_ids(line, self.ph_vocab) for line in self.test_ph]
        self.test_gr_ids = [self.symbols_to_ids(line, self.gr_vocab) for line in self.test_gr]
        
        self.train_gr_length = sorted(max(self.train_gr_ids, key=len), reverse=True)[0]
        self.valid_gr_length = sorted(max(self.valid_gr_ids, key=len), reverse=True)[0]
        self.test_gr_length = sorted(max(self.test_gr_ids, key=len), reverse=True)[0]

        self.train_ph_length = sorted(max(self.train_ph_ids, key=len), reverse=True)[0]
        self.valid_ph_length = sorted(max(self.valid_ph_ids, key=len), reverse=True)[0]
        self.test_ph_length = sorted(max(self.test_ph_ids, key=len), reverse=True)[0]

        self.max_input_length = max(self.train_gr_length, self.valid_gr_length, self.test_gr_length)
        self.max_output_length = max(self.train_ph_length, self.valid_ph_length, self.test_ph_length)
        datasets = (self.train_gr_ids, 
                    self.train_ph_ids, 
                    self.valid_gr_ids, 
                    self.valid_ph_ids, 
                    self.test_gr_ids, 
                    self.test_ph_ids)
        vocabs = (self.gr_vocab, self.ph_vocab)
        self.gr_size = len(self.gr_vocab)
        self.ph_size = len(self.ph_vocab)
        
        params = (self.max_input_length, self.max_output_length, )

    
    

    def getTrain(self, padded=False, one_hot=False):
        ''' Return training set X and y
            Prepare and format the dataset for different types of trining,
            as a vector of integer values or a 3D array with one hot encoding.
            Args:
            padded: (default: False) whether this method returns a padded array or not.
            one_hot: (default: False) this only works if padded=True, if True, returns
                    the dataset as an one-hot encoded array
            Return: Dictionary with input data in "X" key and output data in "y" key

        '''
        X_data = []
        y_data = []
        if padded:
            X_data = sequence.pad_sequences(self.train_gr_ids, maxlen=self.max_input_length)
            X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))
            y_data = sequence.pad_sequences(self.train_ph_ids, maxlen=self.max_output_length)
            y_data = np.reshape(y_data, (y_data.shape[0], y_data.shape[1],1))
            if one_hot:
                X_data = np.array([self.onehot(x, self.gr_size) for x in X_data])
                y_data = np.array([self.onehot(y, self.ph_size) for y in y_data])
                
        else:
            X_data = self.train_gr_ids
            y_data = self.train_ph_ids
        return {"X": X_data, "y": y_data}
    def getValid(self, padded=False, one_hot=False):
        ''' Return validation set X and y
            Prepare and format the dataset for different types of trining,
            as a vector of integer values or a 3D array with one hot encoding.
            Args:
            padded: (default: False) whether this method returns a padded array or not.
            one_hot: (default: False) this only works if padded=True, if True, returns
                    the dataset as an one-hot encoded array
            Return: Dictionary with input data in "X" key and output data in "y" key

        '''
        X_data = []
        y_data = []
        if padded:
            X_data = sequence.pad_sequences(self.valid_gr_ids, maxlen=self.max_input_length)
            X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))
            y_data = sequence.pad_sequences(self.valid_ph_ids, maxlen=self.max_output_length)
            y_data = np.reshape(y_data, (y_data.shape[0], y_data.shape[1],1))
            if one_hot:
                X_data = np.array([self.onehot(x, self.gr_size) for x in X_data])
                y_data = np.array([self.onehot(y, self.ph_size) for y in y_data])
                
        else:
            X_data = self.valid_gr_ids
            y_data = self.valid_ph_ids
        return {"X": X_data, "y": y_data}
    def getTest(self, padded=False, one_hot=False):
        ''' Return test set X and y
            Prepare and format the dataset for different types of trining,
            as a vector of integer values or a 3D array with one hot encoding.
            Args:
            padded: (default: False) whether this method returns a padded array or not.
            one_hot: (default: False) this only works if padded=True, if True, returns
                    the dataset as an one-hot encoded array
            Return: Dictionary with input data in "X" key and output data in "y" key

        '''
        X_data = []
        y_data = []
        if padded:
            X_data = sequence.pad_sequences(self.test_gr_ids, maxlen=self.max_input_length)
            X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))
            y_data = sequence.pad_sequences(self.test_ph_ids, maxlen=self.max_output_length)
            y_data = np.reshape(y_data, (y_data.shape[0], y_data.shape[1],1))
            if one_hot:
                X_data = np.array([self.onehot(x, self.gr_size) for x in X_data])
                y_data = np.array([self.onehot(y, self.ph_size) for y in y_data])
                
        else:
            X_data = self.test_gr_ids
            y_data = self.test_ph_ids
        return {"X": X_data, "y": y_data}