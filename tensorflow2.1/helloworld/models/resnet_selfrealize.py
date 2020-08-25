
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

from tensorflow.keras import Model
# 构建ResNetBlock的class
class ResnetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path
        
        # 第1个部分
        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        
        # 第2个部分
        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()
        
        self.a2 = Activation('relu')

    def call(self, inputs):
        # residual等于输入值本身，即residual=x
        residual = inputs  
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)
        
        # 如果维度不同则调用代码，否则不执行
        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)
        
        # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        out = self.a2(y + residual)  
        return out

# 构建ResNetBlock的class
class ResnetBottleBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBottleBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path
        
        # 第1个部分
        self.c1 = Conv2D(filters, (1, 1), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        
        # 第2个部分
        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()
        
        # 第3个部分
        self.c1 = Conv2D(4*filters, (1, 1), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(4*filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()
        self.a2 = Activation('relu')

    def call(self, inputs):
        # residual等于输入值本身，即residual=x
        residual = inputs  
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)
        
        x = self.c3(x)
        y = self.b3(x)
        
        # 如果维度不同则调用代码，否则不执行
        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)
        
        # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        out = self.a2(y + residual)  
        return out


class ResNet(Model):

    def __init__(self,blockFunc, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        # 结构定义
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = blockFunc(self.out_filters, strides=2, residual_path=True)
                else:
                    block = blockFunc(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet 
                
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
            
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y
    
def resnet_18():
    return ResNet( ResnetBlock, block_list = [2, 2, 2, 2]);

def resnet_34():
    return ResNet( ResnetBlock, block_list = [3, 4, 6, 3]);

def resnet_50():#not pass
    return ResNet( ResnetBottleBlock, block_list = [3, 4, 6, 3]);


resnet_18()

    