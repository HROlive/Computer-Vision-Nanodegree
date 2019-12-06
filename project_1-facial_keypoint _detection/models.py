import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        self.conv1 = nn.Conv2d(1, 32, stride=1, kernel_size=5, padding = 2)
        self.conv2 = nn.Conv2d(32, 64, stride=1, kernel_size=5, padding = 2)
        self.conv3 = nn.Conv2d(64, 128, stride=1, kernel_size=3, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, stride=1, kernel_size=3, padding = 1)
        self.conv5 = nn.Conv2d(256, 512, stride=1, kernel_size=3, padding = 1)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 136)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        
        # Define batch normalization
        self.batch_norm32 = nn.BatchNorm2d(32)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.batch_norm256 = nn.BatchNorm2d(256)
        
        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                I.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    I.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                I.xavier_uniform_(m.weight)
                I.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                I.constant_(m.weight, 1)
                I.constant_(m.bias, 0)
        
        # Weights initialization (only works on pytorch 0.4.1+)
        # https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         I.kaiming_uniform_(m.weight)
        #         if m.bias is not None:
        #             I.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         I.xavier_uniform_(m.weight)
        #         I.zeros_(m.bias)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         I.ones_(m.weight)
        #         I.zeros_(m.bias)
                
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as
        # dropout or batch normalization) to avoid overfitting
        
   
    def forward(self, x):
        ## Define the feedforward behavior of this model
        
        # input (1, 224, 224)
        x = self.max_pool(F.relu(self.batch_norm32(self.conv1(x))))
        # input (32, 112, 112)
        x = self.max_pool(F.relu(self.batch_norm64(self.conv2(x))))
        # input (64, 56, 56)
        x = self.max_pool(F.relu(self.batch_norm128(self.conv3(x))))
        # input (128, 28, 28)
        x = self.max_pool(F.relu(self.batch_norm256(self.conv4(x))))
        # input (256, 14, 14)
        x = self.max_pool(F.relu(self.conv5(x)))
        # input (512, 7, 7)
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
