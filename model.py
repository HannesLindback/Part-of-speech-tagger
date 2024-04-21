import torch
import torch.nn as nn

class RNNTagger(nn.Module):

    def __init__(self, word_embedding_dim, hidden_dim, vocabulary_size, 
                 tagset_size, token2idx, rnn_type, bidirectional, dropout, num_layers):
        """An LSTM based tagger

        word_embedding_dim
            The dimensionality of the word embedding
        hidden_dim
            The dimensionality of the hidden state in the LSTM
        vocabulary_size
            The number of unique tokens in the word embedding (including <PAD> etc)
        tagset_size
            The number of unique POS tags (not including <PAD>, as we don't want to predict it)
        token2idx
            The mapping between tokens and their indices.
        rnn_type
            The type of RNN to be used. Can be either GRU or LSTM. Default is LSTM
        bidirectional
            Bool indicating whether a bidirectional layer should be used. 
        dropout
            Float specifying the amount of dropout.
        num_layers
            Int specifying the number of layers.
        """
        
        super(RNNTagger, self).__init__()                                        
        self.hidden_dim_ = hidden_dim                                    
        self.vocabulary_size_ = vocabulary_size
        self.tagset_size_ = tagset_size
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.token2idx = token2idx

        self._word_embedding = nn.Embedding(num_embeddings=vocabulary_size,       
                                                embedding_dim=word_embedding_dim, 
                                                padding_idx=self.token2idx['<PAD>'])
        
        assert rnn_type == 'lstm' or rnn_type == 'gru', "layer_type must be either lstm or gru"

        if rnn_type == 'lstm':
            self._rnn = nn.LSTM(input_size=word_embedding_dim,                   
                                hidden_size=hidden_dim,                        
                                batch_first=True, bidirectional=bidirectional, 
                                num_layers=num_layers)
        elif rnn_type == 'gru':
            self._rnn = nn.GRU(input_size=word_embedding_dim,
                                hidden_size=hidden_dim,
                                batch_first=True, bidirectional=bidirectional, 
                                num_layers=num_layers, dropout=dropout)
        
        self._fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, tagset_size)

        self._softmax = nn.LogSoftmax(dim=1)                          

        self.training_loss_ = list()
        self.training_accuracy_ = list()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, padded_sentences):
        """The forward pass through the network"""

        batch_size, max_sentence_length = padded_sentences.size()
        embedded_sentences = self._word_embedding(padded_sentences)
        sentence_lengths = (padded_sentences!=self.token2idx['<PAD>']).sum(dim=1)
        sentence_lengths = sentence_lengths.long().cpu()
        X = nn.utils.rnn.pack_padded_sequence(embedded_sentences, sentence_lengths,
                                            batch_first=True, enforce_sorted=False)
        
        rnn_output, _ = self._rnn(X)
        
        X, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        dropout_output = self.dropout(X)
        X = X.contiguous().view(-1, dropout_output.shape[2])

        tag_space = self._fc(X)
        tag_scores = self._softmax(tag_space)
        return tag_scores.view(batch_size, max_sentence_length, self.tagset_size_)
