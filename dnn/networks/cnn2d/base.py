import torch
import torch.nn as nn
import torch.nn.functional as F
    

from base.base_net import BaseNet


class base_CNN_2D_Encoder(BaseNet):
    def __init__(self):
        super(base_CNN_2D_Encoder, self).__init__()
        
        self.rep_dim = 160 # change this part 
       
        #Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d((2,2), return_indices=True)
        self.linear1 = nn.Linear(256 * 12 * 12, 1024)
        self.linear2 = nn.Linear(1024, self.rep_dim)

        self.act = nn.Softsign()


    def forward(self, x, return_indices=False):
        
        # torch input [batch, channel, height, width]
        indices = []
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 2)
        x = torch.transpose(x, 1, 2)
        x=  torch.reshape(x, (-1, 1, 48, 48))
        # x=  torch.reshape(x, (-1, 1, 64, 20))
        x += self.init_bias

        if self.input_sigmoid : 
            x = F.sigmoid(1/x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x, index = self.pool(x)
        indices.append(index)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x, index = self.pool(x)
        indices.append(index)
        x = self.act(x)
        # print(x.shape)

        x = torch.flatten(x, start_dim = 1)
        # print(x.shape)

        x = self.linear1(x)
        x = self.linear2(x)

        if return_indices : 
            return torch.squeeze(x), indices
        else : 
            return torch.squeeze(x)

class base_CNN_2D_Decoder(BaseNet):
    def __init__(self):
        super(base_CNN_2D_Decoder, self).__init__()

        self.rep_dim = 160

        #Decoder
        self.linear1 = nn.Linear(self.rep_dim, 1024)
        self.linear2 = nn.Linear(1024, 256 * 12 * 12)
        self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)

        self.t_conv2 = nn.ConvTranspose2d(128, 64, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)

        self.t_conv3 = nn.ConvTranspose2d(64, 32, 3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)

        self.t_conv4 = nn.ConvTranspose2d(32, 1, 3, padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(1)

        self.unpool = nn.MaxUnpool2d((2,2))

        self.act = nn.Softsign()
    

    def forward(self, x, indices):

        x = self.linear1(x)
        x = self.linear2(x)

        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 2)

        x = torch.reshape(x, (-1, 256, 12, 12))
        
        x = self.unpool(x, indices[-1])
        x = self.t_conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.t_conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.unpool(x, indices[-2])
        x = self.t_conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.t_conv4(x)
        x = self.bn4(x)
        x = torch.flatten(x, start_dim = 1)
        x = F.sigmoid(x)
    
        return torch.squeeze(x)


class base_CNN_2D_Autoencoder(BaseNet):
    def __init__(self):
        super(base_CNN_2D_Autoencoder, self).__init__()

        self.rep_dim= 160

        self.encoder = base_CNN_2D_Encoder()
        self.decoder = base_CNN_2D_Decoder()

    def forward(self, x):

        x, indices = self.encoder(x, return_indices=True)
        x = self.decoder(x, indices)
                  
        return x






