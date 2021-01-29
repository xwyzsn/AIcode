import numpy as np
import random
import math
from tensorflow.python.keras import layers
from tensorflow.python.keras import Sequential
import tensorflow as tf
from tqdm import tqdm
class FNN:
    # 初始化
    def __init__(self, train_data, train_label, hidden_num, output_num):
        self.train_data = train_data
        # print 'train_data',self.train_data,train_label
        self.train_label = train_label
        self.hidden_num = hidden_num
        self.outpu_num = output_num
        # 隐藏层的数据
        self.hidden_data = [0 for i in range(hidden_num)]
        # 输出层的数据
        self.output_data = [0 for i in range(output_num)]
        # 输入层到隐藏层的weight
        self.i_h_weight = [[random.uniform(-1.0, 1.0) for j in range(hidden_num)] for i in range(len(train_data[0]))]
        # 隐藏层到输出层的weight
        self.h_o_weight = [[random.uniform(-1.0, 1.0) for i in range(output_num)] for j in range(hidden_num)]
        # 隐藏层的b
        self.hidde_b = [0 for i in range(hidden_num)]
        # 输出层的b
        self.output_b = [0 for j in range(output_num)]
        self.error = [0 for i in range(output_num)]

    # sigmod激活函数
    def sigmod(self, x):
        return 1.0 / (1 + math.exp(-x))
    #sigmoid 求导
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, input_data):
        total = []
        for i in range(self.hidden_num):
            result = 0
            for j in range(len(input_data)):
                result += input_data[j] * self.i_h_weight[j][i]
            total.append(self.sigmod(result))
        final = []
        for i in range(self.outpu_num):
            r2 = 0
            for j in range(self.hidden_num):
                r2 += total[j] * self.h_o_weight[j][i]
            final.append(self.sigmod(r2))
        out_lable = []
        for i in final:
            max_arg = np.array(i).argmax()
            out_lable.append(max_arg)
        return final

    # 前馈网络
    def feedforward(self, data_index):
        # 隐藏层的数据
        self.hidden_data = [0 for i in range(self.hidden_num)]
        # 输出层的数据
        self.output_data = [0 for i in range(self.outpu_num)]
        # print self.hidden_data
        # 只对一条数据进行训练的
        # 得到隐藏层的数据
        for i in range(self.hidden_num):
            total = 0.0
            for j in range(len(self.train_data[0])):
                total += self.train_data[data_index][j] * self.i_h_weight[j][i]
            total += self.hidde_b[i]
            self.hidden_data[i] = self.sigmod(total)
        # 得到输出层的数据
        for i in range(self.outpu_num):
            total = 0.0
            for j in range(self.hidden_num):
                total += self.hidden_data[j] * self.h_o_weight[j][i]
            total += self.output_b[i]

            self.output_data[i] = self.sigmod(total)
        return self.output_data[0]

    # BP反馈网络
    def feedback(self, MM, data_index):
        # 前馈网络
        self.feedforward(data_index)

        # #更新隐藏层到输出层的weight和b
        for i in range(len(self.output_data)):
            # 求导后的就是两个做差
            self.error[i] = self.train_label[data_index][i] - self.output_data[i]


        for i in range(self.outpu_num): #遍历每个out节点
            for j in range(self.hidden_num):# 更新每个hidden的节点对于i 的误差
                # 权重 : = 权重+学习率*导数
                self.h_o_weight[j][i] += MM * self.hidden_data[j] * self.error[i] * self.output_data[i] * (
                            1 - self.output_data[i])
            self.output_b[i] += MM * self.output_data[i] * (1 - self.output_data[i]) * self.error[i]
        # 更行输入层到输出层的weight和b
        for i in range(self.hidden_num):#遍历每个hidden节点
            sum_ek = 0.0
            for k in range(self.outpu_num):
                #计算每个input的节点对于该节点的误差
                sum_ek += self.h_o_weight[i][k] * self.error[k] * self.output_data[k] * (1 - self.output_data[k])
            for j in range(len(self.train_data[0])):
                self.i_h_weight[j][i] += MM * self.hidden_data[i] * (1 - self.hidden_data[i]) * \
                                         self.train_data[data_index][j] * sum_ek

            self.hidde_b[i] += MM * self.hidden_data[i] * (1 - self.hidden_data[i]) * sum_ek

    # 训练
    def train(self, train_num, MM ):
        for i in tqdm(range(train_num)):
            #这里的10是hiddenlayer中有10个units
            for j in range(10):
                self.feedback(MM, j)



def Keras_model():
    # tensorflow一个BP网络
    model = Sequential()
    model.add(layers.Dense(63))
    model.add(layers.Dense(10, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))#多分类用softmax
    model.compile(optimizer='adam', loss=tf.losses.categorical_crossentropy,
                  metrics=tf.metrics.categorical_accuracy)
    model.fit(train, train_label, epochs=500)
    print(model.predict_classes(test))
    print(model.predict(test))


def my_model(epoch=1000,learning_rate=0.3):
    global index, i
    object = FNN(train, train_label, 5, 10)
    object.train(epoch, learning_rate)
    # print("第一层权重")
    # print(np.array(object.i_h_weight).reshape(-1,5))
    # print("第二层权重")
    # print(np.array(object.h_o_weight).reshape(-1,10))
    l=[]
    for item in test:
        out = object.predict(item)
        out_put = []
        for k, i in enumerate(out):
            if i > 0.9:
                i = 1
            elif i < 0.1:
                i = 0
            else:
                i = i
            out_put.append((k, i))
        # print("取值可能性:\n",out_put)
        # print("预测为: ",np.array(object.predict(item)).argmax())
        l.append(np.array(object.predict(item)).argmax())
    return l



if __name__ == '__main__':
    #train_data
    train = [ \
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 1, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
    ]
   #test_data手打真的好累
    test = [ \
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 0, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 1, 0, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 0, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 1, 0, 0,
         0, 0, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
        ,
        [0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0
         ]
    ]
   #lable 为一个向量,对应的数值,该位置为1
    train_label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    print("keras_model")
    Keras_model()
    print("my_model")
    my_model()
    #不同epoch
    print("epoch=100",my_model(100),"\n")
    print("epoch=1000",my_model(1000),"\n")
    print("epoch=5000",my_model(5000),"\n")

    #不同的学习率
    print("learning_rate=0.01",my_model(learning_rate=0.01),"\n")
    print("learning_rate=0.3",my_model(learning_rate=0.3),"\n")
    print("learning_rate=0.8",my_model(learning_rate=0.8),"\n")