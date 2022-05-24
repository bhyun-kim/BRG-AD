import torch
import torch.nn as nn
import torch.nn.functional as F
    
from base.base_net import BaseNet


class fullyConv_1D_Encoder(BaseNet):
    def __init__(self):
        super(fullyConv_1D_Encoder, self).__init__()
        
        self.rep_dim = 160 
       
        #Encoder
        self.conv1 = nn.Conv1d(1, 32, 3 * 3 * 3, padding=13)  
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 3 * 3 * 3, padding=13)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, 3 * 3 * 3, padding=13)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, 3 * 3 * 3, padding=13)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 160, 9)

        self.pool = nn.MaxPool1d((4), return_indices=True)

        self.act = nn.Softsign()


    def forward(self, x, return_indices=False):
        
        indices = []

        x_res = x[:, :, -2304:]
        

        x = self.conv1(x_res)
        x = self.bn1(x)
        # print(x.shape)
        # x = torch.cat((self.act(x), x_res.repeat(1,16,1)), dim=1)
        x = self.act(x)
        # print(x.shape)
        x_res, index = self.pool(x)
        # print(x.shape)
        indices.append(index)


        x = self.conv2(x_res)
        x = self.bn2(x)
        x = self.act(x)
        x_res, index = self.pool(x)
        indices.append(index)
        

        x = self.conv3(x_res)
        x = self.bn3(x)
        x = self.act(x)
        x_res, index = self.pool(x)
        indices.append(index)   

        x = self.conv4(x_res)
        x = self.bn4(x)
        x = self.act(x)
        x_res, index = self.pool(x)
        indices.append(index)

        # print(x_res.shape)
        
        x = self.conv5(x_res)
        

        if return_indices : 
            return torch.squeeze(x), indices
        else : 
            return torch.squeeze(x)

class fullyConv_1D_Decoder(BaseNet):
    def __init__(self):
        super(fullyConv_1D_Decoder, self).__init__()

        self.rep_dim = 160

        #Decoder
        self.linear1 = nn.Linear(self.rep_dim, 1024)
        self.linear2 = nn.Linear(1024, 2304)
        self.t_conv1 = nn.ConvTranspose1d(256, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.t_conv2 = nn.ConvTranspose1d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.t_conv3 = nn.ConvTranspose1d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)

        self.t_conv4 = nn.ConvTranspose1d(32, 1, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(1)

        self.unpool = nn.MaxUnpool1d(4)

        self.act = nn.Softsign()
    

    def forward(self, x, indices):

        x = self.linear1(x)
        x = self.linear2(x)

        x = torch.unsqueeze(x, 2)
        x = torch.transpose(x , 1, 2)

        x = x.view(x.size(0), 256, 9)
        
        x = self.unpool(x, indices[-1])
        x = self.t_conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.unpool(x, indices[-2])
        x = self.t_conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.unpool(x, indices[-3])
        x = self.t_conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.unpool(x, indices[-4])
        x = self.t_conv4(x)
        x = self.bn4(x)
        x = F.sigmoid(x)
    
        return torch.squeeze(x)


class fullyConv_1D_Autoencoder(BaseNet):
    def __init__(self):
        
        super(fullyConv_1D_Autoencoder, self).__init__()

        self.rep_dim= 160
        self.encoder = fullyConv_1D_Encoder()
        self.decoder = fullyConv_1D_Decoder()

    def forward(self, x):

        x, indices = self.encoder(x, return_indices=True)
        x = self.decoder(x, indices)
                  
        return x






