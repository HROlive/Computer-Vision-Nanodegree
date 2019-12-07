import torch
import torch.nn as nn
import torchvision.models as models
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        # linear layer weights initialization
        I.xavier_uniform_(self.embed.weight)
        I.constant_(self.embed.bias, 0)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize the PyTorch RNN Module
        :param embed_size: The size of embeddings, should you choose to use them        
        :param hidden_size: The size of the hidden layer outputs
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param num_layers: Number of LSTM/GRU layers
        """
        super(DecoderRNN, self).__init__()
        
        # set class variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # define embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # define the fully connected layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                I.uniform_(m.weight, -1, 1)
            elif isinstance(m, nn.Linear):
                I.xavier_uniform_(m.weight)
                I.constant_(m.bias, 0)
    
    def forward(self, features, captions):
        # remove the <end> token to avoid predicting when it's the input to the LSTM
        captions = captions[:, :-1] # (batch_size, caption_length) -> (batch_size, captions_length - 1)
        
        # create embedded word vectors
        word_embeds = self.embedding(captions) # (batch_size, caption_length - 1, embed_size)
        
        # concatenate features and captions
        inputs = torch.cat((features.unsqueeze(1), word_embeds), dim=1) # (batch_size, caption length , embed_size)
        
        # first value returned by LSTM -> all hidden states
        # second value returned by LSTM -> most recent hidden state
        lstm_output, _ = self.lstm(inputs) # (batch_size, caption length, hidden_size)
        
        # get the word scores
        outputs = self.linear(lstm_output) # (batch_size, caption length, vocab_size)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # features shape (1, 1, embed_size)
        caption = []
        
        for i in range(max_len):
            lstm_output, last_state = self.lstm(inputs, states) # (1, 1, embed_size)
            output = self.linear(lstm_output).squeeze(1) # (1, 1, vocab_size) -> (1, vocab_size)
            word_idx = output.argmax(dim=1)
            caption.append(word_idx.item())
            
            if word_idx == 1:
                break
            
            inputs = self.embedding(word_idx.unsqueeze(1))
            states = last_state
        
        return caption