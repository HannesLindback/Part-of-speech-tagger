import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from preprocessing import Utils, Data, Encode, ToTensor
from model import RNNTagger

class Tagger:

    def __init__(self, model=None, word_embedding_dim=64, hidden_dim=128, 
                 token_mapping=None, label_mapping=None, n_epochs=5, 
                 batch_size=256, rnn_type='gru', learning_rate=0.01, bidirectional=False, 
                 dropout=0, num_layers=1, weight_decay=0, masking=0):
        """Class providing an API for fitting, scoring and predicting with an RNN-based tagger
        
        Args:
            model: The model used by the tagger.
            word_embedding_dim: The dimensionality of the word embedding
            hidden_dim: The dimensionality of the hidden state in the LSTM
            token_mapping: A mapping between tokens and their indices.
            labels_mapping: A mapping between labels and their indices. 
            min_epochs: The number of epochs for fitting the model to the data.
            batch_size: The size of the minibatches.
            rnn_type: The type of RNN to be used. Can be either GRU or LSTM. Default is LSTM.
            learning_rate: The initial learning rate. Default is 0.01.
            bidirectional: Bool indicating whether a bidirectional layer should be used. 
            dropout: Float specifying the amount of dropout.
            num_layers: Int specifying the number of layers.
            weight_decay: The decay of the weights in the optimizer.
            masking: The amount of masking of tokens during training.
        """

        self.model = model
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.utils = Utils()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.lr = learning_rate
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.min_epochs = n_epochs
        self.batch_size = batch_size
        self.masking = masking
        self.token2idx = token_mapping
        self.label2idx = label_mapping

        self.training_loss_ = list()                                  
        self.training_accuracy_ = list()

    def fit(self, data):
        """Fits the model after the provided data.
        
        Argument data must be either a path to a conllu-file, a list of lists 
        with tuples of word-tag pairs or a pytorch Dataset."""

        if isinstance(data, str) and data[-6:] == 'conllu':
            dataset = self._transform(path=data, train=True)
        elif isinstance(data, Dataset):
            dataset = data
        else:
            dataset = self._transform(labeled_sentences=data, train=True)

        if self.model is None:
            self.model = RNNTagger(word_embedding_dim=self.word_embedding_dim, 
                                   hidden_dim=self.hidden_dim,
                                   vocabulary_size=len(self.token2idx), 
                                   tagset_size=len(self.label2idx)-1, 
                                   token2idx=self.token2idx, rnn_type=self.rnn_type, 
                                   bidirectional=self.bidirectional, dropout=self.dropout, 
                                   num_layers=self.num_layers)
            
        loss_function = nn.NLLLoss(ignore_index=self.label2idx['<PAD>'])               
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)           

        for epoch in range(self.min_epochs):
            with tqdm(DataLoader(dataset, batch_size=self.batch_size, shuffle=False), 
                    total=len(dataset.X)//self.batch_size+1, unit="batch", 
                    desc="Epoch %i" % epoch) as batches:
                
                for batch in batches:
                    loss, accuracy = self._fit(batch, loss_function, optimizer)
                    batches.set_postfix(loss=loss.item(), accuracy=accuracy)

    def _fit(self, batch, loss_function, optimizer):
        """Performs one forward/backward pass on one batch of training data.
        
        Returns the loss and accuracy for batch."""

        inputs, targets = self.utils.remove_extra_padding(batch)
        self.model.zero_grad()

        scores = self.model(inputs)

        loss = loss_function(scores.view(-1, self.model.tagset_size_), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        predictions = scores.argmax(dim=2, keepdim=True).squeeze()
        mask = targets!=self.label2idx['<PAD>']
        correct = (predictions[mask] == targets[mask]).sum().item()

        accuracy = correct / mask.sum().item()*100
        self.training_accuracy_.append(accuracy)
        self.training_loss_.append(loss.item())

        return loss, accuracy
        
    def predict(self, sentence):
        """Predicts POS-tags for a sentence.
        
        Tokenizes sentence if string.
        Returns a list of tuples with word,tag."""

        if isinstance(sentence, str):
            sentence = word_tokenize(sentence)

        encoded_sentence = self._transform(sentences=sentence)

        for batch in DataLoader(encoded_sentence, batch_size=1, shuffle=False):
            inputs, _ = self.utils.remove_extra_padding(batch)
            scores = self.model(inputs)         
            predictions = scores.argmax(dim=2, keepdim=True).squeeze()

        return self.utils.decode(inputs, predicted_labels=predictions)

    def score(self, data, test_batch_size=1):
        """Returns the accuracy of the model.
        
        Argument data must be either a path to a conllu-file, a list of lists 
        with tuples of word-tag pairs or a pytorch Dataset."""

        if isinstance(data, str) and data[-6:] == 'conllu':
            dataset = self._transform(path=data, train=False)
        elif isinstance(data, Dataset):
            dataset = data
        else:
            dataset = self._transform(labeled_sentences=data, train=True)

        with torch.no_grad(): 
            n_correct = 0
            n_total = 0

            for batch in DataLoader(dataset, batch_size=test_batch_size, shuffle=False):
                inputs, targets = self.utils.remove_extra_padding(batch)
                scores = self.model(inputs)
                predictions = scores.argmax(dim=2, keepdim=True).squeeze()

                # Reshape predictions so the size is compatible with targets.
                if test_batch_size == 1:
                    predictions = torch.reshape(predictions, (1, predictions.shape[0]))
                
                mask = targets!=self.label2idx['<PAD>'] 

                n_correct += (predictions[mask] == targets[mask]).sum().item()  
                n_total += mask.sum().item()

        return n_correct/n_total
        
    def _transform(self, sentences=None, labeled_sentences=None, path=None, train=False):
        """Preprocesses raw data into encoded tensors. 
        
        Will create a new token/tag-index mapping if train is True.
        If train is false and the tagger does not have a token/tag-index mapping
        a token/tag-index mapping will have to be given to the model.

        Can take as argument either a path to a conllu-file, a list of labeled sentences,
        a list of unlabeled sentences.

        Returns:
            ud_data: A pytorch Dataset of a Universal Dependencies treebank tokens and tags.
        """

        assert sentences is not None or path is not None or labeled_sentences is not None, "Arguments sentences, labeled_sentences and path cannot both be None!"
        
        # Extracts sentences and labels from the data in different ways depending 
        # on the structure of the given data.

        if path is None:
            if labeled_sentences is None:
                # If no labels are provided the '<PAD>' label will be used as placeholders.
                labels = ['<PAD>' for i in range(len(sentences))]
            else:
                sentences, labels = [], []
                for labeled_sentence in labeled_sentences:
                    sentence, tags = [], []
                    for word, label in labeled_sentence:
                        sentence.append(word)
                        tags.append(label)
                    sentences.append(sentence)
                    labels.append(tags)
        else:
            sentences, labels = self.utils(path)
        
        padding_size = self.utils.get_padding_size(sentences)

        if train:
            self.token2idx, self.label2idx = self.utils.map(sentences, labels)

        ud_data = Data(X=sentences, y=labels, 
                        transform=transforms.Compose([Encode(padding_size,
                                                             self.token2idx, 
                                                             self.label2idx,
                                                             self.masking),
                                                      ToTensor()]))
        return ud_data
    
    def analyze_sentence(self, labeled_sentence):
        """Function for easier analysis of classified sentences. 

        Prints word, correct label and predicted label in a sentence."""

        print('Word:\tCorrect label:\tPredicted label:')
        for (word, correct, predicted) in labeled_sentence:
            print("%s\t\t%s\t\t%s" % (word, correct, predicted))

def plot(data, tagger):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    ax.set_title("Plot for the (hopefully) decreasing loss over epochs")
    ax.plot(tagger.training_loss_, 'b-')
    ax.set_ylabel("Training Loss", color='b')
    ax.set_xlabel("Epoch")
    # ax.set_yscale('log')
    ax.tick_params(axis='y', labelcolor='b')
    ax = ax.twinx()
    ax.plot(tagger.training_accuracy_, 'r-')
    ax.set_ylabel("Accuracy [%]", color='r')
    ax.tick_params(axis='y', labelcolor='r')
    a = list(ax.axis())
    a[2] = 0
    a[3] = 100
    ax.axis(a)
    t = np.arange(0, len(tagger.training_accuracy_), len(data)//tagger.batch_size+1)
    ax.set_xticks(ticks=t)
    ax.set_xticklabels(labels=np.arange(len(t)))
    fig.tight_layout()
    plt.show()

def baseline():
    """Provides a baseline accuracy of the model.
    
    Uses a simple model with 16 dimensions for the word embedding and 32 hidden dimensions.
    Additionally the model is unidirectional with 1 layer, 0 dropout, weight decay and masking.

    The dataset used is UD English_LinES; a small, but rather balanced corpus.

    10 identical models are created and evaluated on the dataset.
    The baseline accuracy is the average accuracy of the 10 models on the dataset."""
    
    print('Establishing baseline...')

    train_path = 'ud-treebanks/UD_English-LinES/en_lines-ud-train.conllu'
    test_path = 'ud-treebanks/UD_English-LinES/en_lines-ud-test.conllu'

    baseline_accuracy = 0
    for i in range(10):
        model = Tagger(word_embedding_dim=32, hidden_dim=64,
                                rnn_type='lstm', bidirectional=False, 
                                dropout=0, num_layers=1, weight_decay=0, masking=0)
        model.fit(train_path)
        baseline_accuracy += model.score(test_path)

    print("Baseline accuracy %.1f%%" % (100*baseline_accuracy/10))

def dimensionality_testing():
    """Helper function for testing different dimensionality of the tagger.
    
    A "simple model" refers in this case to a unidirectional LSTM network with 0 dropout, masking 
    and weight decay and 1 layer."""
    
    configuration_testing = None

    # Test a simple model with default number of dimensions.
    accuracy = configuration_testing(n_epochs=3, rnn_type='lstm', bidirectional=False, 
                                     dropout=0, num_layers=1, weight_decay=0, masking=0)
    print(f'Average accuracy for simple model with default number of dimensions: {accuracy}')
    
    # Test a simple model with 16 dimension for the embedding and 32 hidden dimensions.
    accuracy = configuration_testing(word_embedding_dim=16, hidden_dim=32, 
                                     n_epochs=3, rnn_type='lstm', bidirectional=False, 
                                     dropout=0, num_layers=1, weight_decay=0, masking=0)
    print(f'Average accuracy for simple model with lower number of dimensions: {accuracy}')
    
    # Test a simple model with 64 dimension for the embedding and 128 hidden dimensions.
    accuracy = configuration_testing(word_embedding_dim=64, hidden_dim=128, 
                                     n_epochs=3, rnn_type='lstm', bidirectional=False, 
                                     dropout=0, num_layers=1, weight_decay=0, masking=0)
    print(f'Average accuracy for simple model with higher number of dimensions: {accuracy}')
    
    # Test a simple model with 128 dimension for the embedding and 256 hidden dimensions.
    accuracy = configuration_testing(word_embedding_dim=128, hidden_dim=256, 
                                     n_epochs=3, rnn_type='lstm', bidirectional=False, 
                                     dropout=0, num_layers=1, weight_decay=0, masking=0)
    print(f'Average accuracy for simple model with much higher number of dimensions: {accuracy}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from preprocessing import Utils

    train_path = 'ud-treebanks/UD_English-PUD/en_pud-ud-train.conllu'
    test_path = 'ud-treebanks/UD_English-PUD/en_pud-ud-test.conllu'
    'ud-treebanks\UD_English-PUD\en_pud-ud-test.conllu'

    tagger = Tagger(rnn_type='gru', bidirectional=True, dropout=0.1, num_layers=1, weight_decay=0, masking=0)
    tagger.fit(data=train_path)
    print(tagger.score(data=test_path))
    
