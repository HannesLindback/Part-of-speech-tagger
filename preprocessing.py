import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random


class Data(Dataset):
    
    def __init__(self, X, y, transform=None) -> None:
        """Used together with class Dataloader to structure and load dataset."""

        # Wrap single sentence inside list.
        if not isinstance(X[0], list) and not isinstance(y[0], list):
            X, y = [X], [y]
        self.X, self.y = X, y
        self.transform  = transform
        

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'X': self.X[idx], 'y': self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class Encode:
    
    def __init__(self, padding_size, token2idx, label2idx, masking):
        """Class for encoding sentences and labels according to their word/label types indices in the dataset."""

        self.padding_size = padding_size
        self.token2idx, self.label2idx = token2idx, label2idx
        self.masking = masking
    
    def __call__(self, sample):
        """Encodes sentences and labels according to their indices.
        
        If a word is unknown it gets encoded as an OOV-word, '<UNK>.
        Masks tokens according to the given probability.
        Adds padding to sentence and labels accoding to self.padding_size."""

        sentence, labels = sample['X'], sample['y']
        encoded_sentence = []
        for i in range(self.padding_size):
            if i < len(sentence):
                if self.token2idx.get(sentence[i], False):
                    # Mask token
                    if random.random() < self.masking:
                        encoded_sentence.append(self.token2idx['<MASK>'])
                    else:
                        encoded_sentence.append(self.token2idx[sentence[i]])
                else:
                    encoded_sentence.append(self.token2idx['<UNK>'])
            else:
                encoded_sentence.append(self.token2idx['<PAD>'])

        encoded_labels = [self.label2idx[labels[i]]
                          if i < len(labels) else self.label2idx['<PAD>']
                          for i in range(self.padding_size)]
        return {'X': encoded_sentence, 'y': encoded_labels}
    

class ToTensor:
    """Converts lists of encoded sentences and labels to tensors."""

    def __call__(self, sample):
        sentence, labels = sample['X'], sample['y']
        sentence = torch.LongTensor(sentence)
        labels = torch.LongTensor(labels)
        
        if torch.cuda.is_available():
            return {'X': sentence.cuda(), 'y': labels.cuda()}
        else:
            return {'X': sentence, 'y': labels}


class Utils:
    
    def __init__(self) -> None:
        """Class containing functions for processing and structuring data."""

        self._token2idx = None
        self._label2idx = None
        self._idx2token = None
        self._idx2tag = None

    def __call__(self, path, tagset='universal'):
        """Loads and reads the tokens and POS-tags from a conllu file.
        
        Returns:
            sentences: A list of lists, each nested list is the tokens of a sentence.
            labels: A list of lists, each nested list is the POS-tags of a sentence.
        """ 
        
        assert path[-6:] == 'conllu', "Path must lead to a .conllu file."

        sentences = []
        labels = []

        with open(path, 'r', encoding='utf-8') as f:
            sent_tokens = []
            sent_pos_tags = []
            for line in f:
                if line == '\n':
                    if len(sent_tokens) > 2:  # Remove short sentences
                        sentences.append(sent_tokens)
                        labels.append(sent_pos_tags)
                    sent_tokens = []
                    sent_pos_tags = []
                elif line[0] != '#':
                    _, token, _, POS_tag, XPOS = line.split('\t')[:5]
                    sent_tokens.append(token)
                    if tagset == 'universal':
                        sent_pos_tags.append(POS_tag)
                    elif tagset == 'language_specific':
                        if XPOS != '_':
                            sent_pos_tags.append(XPOS)
                        else:
                            sent_pos_tags.append(POS_tag)

        print("Loaded %i sentences" % len(sentences))
        assert len(sentences) == len(labels)
        assert np.all([len(sentence)==len(tags) 
                       for sentence, tags in zip(sentences, labels)])
        
        return sentences, labels

    def map(self, sentences, labels):
        """Creates a mapping between tokens and their indices used to create the encoding."""

        if not isinstance(sentences, list) and isinstance(labels, list):
            sentences, labels = [sentences], [labels]

        tokens = {token for sentence in sentences for token in sentence}
        idx2token = list(tokens)
        idx2token.insert(0, '<UNK>')
        idx2token.insert(1, '<MASK>')
        idx2token.append('<PAD>')
        token2idx = {token:idx for idx, token in enumerate(idx2token)}

        tags = {tag for tags in labels for tag in tags}
        idx2tag = list(tags)
        idx2tag.append('<PAD>')
        tag2idx = {tag:idx for idx, tag in enumerate(idx2tag)}

        self._token2idx, self._label2idx = token2idx, tag2idx
        self._idx2token, self._idx2tag = idx2token, idx2tag

        return token2idx, tag2idx

    def get_padding_size(self, sentences):
        """Finds out how much padding should be added to the sentences."""

        return max([len(sentence) for sentence in sentences])

    def remove_extra_padding(self, batch):
        """Removes the extra padding for each batch.
        
        Each batch will have the length of the longest sentence in the batch
        instead of in the entire dataset."""

        inputs, targets = batch['X'], batch['y']
        inputs = inputs[:, :torch.max((inputs!=self._token2idx['<PAD>']).sum(dim=1))]
        targets = targets[:, :torch.max((targets!=self._label2idx['<PAD>']).sum(dim=1))]
        return inputs, targets

    def decode(self, encoded_sentence, encoded_labels=None, predicted_labels=None):
        """Decodes one encoded sentence and labels back to text.
        
        Returns a list of tuples with the labeled sentence."""

        sentence = [self._idx2token[elem] for elem in encoded_sentence[0]]

        if predicted_labels is not None and encoded_labels is not None:
            predicted_labels = [self._idx2tag[elem] for elem in predicted_labels]
            labels = [self._idx2tag[elem] for elem in encoded_labels[0]]
            return list(zip(sentence, labels, predicted_labels))
        elif encoded_labels is None:
            predicted_labels = [self._idx2tag[elem] for elem in predicted_labels]
            return list(zip(sentence, predicted_labels))
        else:
            return list(zip(sentence, labels))
    