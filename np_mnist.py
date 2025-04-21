import numpy as np

import statistics

from tqdm  import tqdm


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    return np.maximum(x, 0)

def relu_prime(x):
    '''
    relu函数的导数
    '''
    return np.where(x > 0, 1, 0)

#输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def f_prime(x):
    '''
    softmax函数的导数
    '''
    return f(x) * (1 - f(x))

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return -np.sum(y_true * np.log(y_pred + 1e-8), axis=-1)

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return y_pred - y_true



# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, lr=0.01, momentum=0.35):
        '''
        初始化网络结构
        '''
        self.lr = lr
        self.momentum = momentum
        self.w1 = init_weights((input_size, hidden_size_1))
        self.b1 = init_weights((hidden_size_1,))
        self.w2 = init_weights((hidden_size_1, hidden_size_2))
        self.b2 = init_weights((hidden_size_2,))
        self.w3 = init_weights((hidden_size_2, output_size))
        self.b3 = init_weights((output_size,))

        self.momentum_w2 = np.zeros_like(self.w2)
        self.momentum_b2 = np.zeros_like(self.b2)
        self.momentum_w1 = np.zeros_like(self.w1)
        self.momentum_b1 = np.zeros_like(self.b1)
        self.momentum_w3 = np.zeros_like(self.w3)
        self.momentum_b3 = np.zeros_like(self.b3)

    def forward(self, x):
        '''
        前向传播
        '''
        self.z1 = np.matmul(x, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.matmul(self.a1, self.w2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.matmul(self.a2, self.w3) + self.b3
        self.a3 = f(self.z3)
        return self.a3

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        batch_size = x_batch.shape[0]

        # 前向传播
        y_pred = self.forward(x_batch)

        # 计算损失和准确率
        loss = loss_fn(y_batch, y_pred)
        acc = np.mean(np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1))

        # 反向传播
        delta_3 = loss_fn_prime(y_batch, y_pred)
        delta_2 = (delta_3 @ self.w3.T) * relu_prime(self.a2)
        delta_1 = (delta_2 @ self.w2.T) * relu_prime(self.a1)

        grads_w3 = self.a2.T @ delta_3 / batch_size
        grads_b3 = np.sum(delta_3, axis=0) / batch_size
        grads_w2 = self.a1.T @ delta_2 / batch_size
        grads_b2 = np.sum(delta_2, axis=0) / batch_size
        grads_w1 = x_batch.T @ delta_1 / batch_size
        grads_b1 = np.sum(delta_1, axis=0) / batch_size


        # 更新权重

        self.momentum_w1 = self.momentum_w1 * self.momentum + (1 - self.momentum) * grads_w1
        self.momentum_w2 = self.momentum_w2 * self.momentum + (1 - self.momentum) * grads_w2
        self.momentum_w3 = self.momentum_w3 * self.momentum + (1 - self.momentum) * grads_w3
        self.momentum_b1 = self.momentum_b1 * self.momentum + (1 - self.momentum) * grads_b1
        self.momentum_b2 = self.momentum_b2 * self.momentum + (1 - self.momentum) * grads_b2
        self.momentum_b3 = self.momentum_b3 * self.momentum + (1 - self.momentum) * grads_b3

        self.w3 -= self.lr * self.momentum_w3
        self.b3 -= self.lr * self.momentum_b3
        self.w2 -= self.lr * self.momentum_w2
        self.b2 -= self.lr * self.momentum_b2
        self.w1 -= self.lr * self.momentum_w1
        self.b1 -= self.lr * self.momentum_b1

        return loss, acc


if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size_1=256, hidden_size_2=256, output_size=10, lr=0.45)
    for epoch in range(10):
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i: i + 64]
            y_batch = y_train[i: i + 64]
            loss, acc = net.step(x_batch, y_batch)
            losses.append(loss)
            accuracies.append(acc)
        print("epoch:{} acc:{:.4f}".format(epoch + 1, statistics.mean(accuracies)))
    
    y_pred = net.forward(X_val)
    val_acc = np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_val, axis=-1))
    print("val_acc:{}".format(val_acc))