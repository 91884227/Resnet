#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet_18(nn.Module):
    def __init__(self):
        super(ResNet_18, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=(7, 7), stride = 2)
                               
        # conv2
        self.conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3), padding = 2)
        self.conv2_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3) )
        self.conv2_3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3), padding = 2)
        self.conv2_4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3) )
        
        # conv3
        self.conv3_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=(3, 3), padding = 2)
        self.conv3_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3, 3)) 
        self.conv3_3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3, 3), padding = 2) 
        self.conv3_4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3, 3)) 
        self.shortcut3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size= 1)
        
        # conv4
        self.conv4_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=(3, 3), padding = 2)
        self.conv4_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3, 3) ) 
        self.conv4_3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3, 3), padding = 2) 
        self.conv4_4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3, 3) ) 
        self.shortcut4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size= 1)
        
        # conv5
        self.conv5_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=(3, 3), padding = 2)
        self.conv5_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3, 3) ) 
        self.conv5_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3, 3), padding = 2) 
        self.conv5_4 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3, 3) ) 
        self.shortcut5 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size= 1)
        
        # other
        self.act_func =  nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2), padding = 0)
        self.avgpool = nn.AvgPool2d(kernel_size = (2, 2), padding = 0)
        self.linear = nn.Linear(512, 5)
    
    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.act_func(x)
        x = self.maxpool(x)
        
        # conv2
        x = self.act_func(self.conv2_2(self.act_func(self.conv2_1(x))) + x) 
        x = self.act_func(self.conv2_4(self.act_func(self.conv2_3(x))) + x)
        x = self.maxpool(x)
        
        # conv3
        x = self.act_func(self.conv3_2(self.act_func(self.conv3_1(x))) + self.shortcut3(x))  
        x = self.act_func(self.conv3_4(self.act_func(self.conv3_3(x))) + x)  
        x = self.maxpool(x)
        
        # conv4
        x = self.act_func(self.conv4_2(self.act_func(self.conv4_1(x))) + self.shortcut4(x))    
        x = self.act_func(self.conv4_4(self.act_func(self.conv4_3(x))) + x)
        x = self.maxpool(x)
        
        # conv5
        x = self.act_func(self.conv5_2(self.act_func(self.conv5_1(x))) + self.shortcut5(x)) 
        x = self.act_func(self.conv5_4(self.act_func(self.conv5_3(x))) + x) 
        x = self.maxpool(x)
        
        # final
        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        
        return(x)

