import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_train = np.array ( [ [ 3.3 ] , [ 4.4 ] , [ 5.5 ] , [ 6.71 ] , [ 6.93 ] , [ 4.168 ] ,
                       [ 9.779 ] , [ 6.182 ] , [ 7.59 ] , [ 2.167 ] , [ 7.042 ] ,
                       [ 10.791 ] , [ 5.313 ] , [ 7.997 ] , [ 3.1 ] ] , dtype=np.float32 )

y_train = np.array ( [ [ 1.7 ] , [ 2.76 ] , [ 2.09 ] , [ 3.19 ] , [ 1.694 ] , [ 1.573 ] ,
                       [ 3.366 ] , [ 2.596 ] , [ 2.53 ] , [ 1.221 ] , [ 2.827 ] ,
                       [ 3.465 ] , [ 1.65 ] , [ 2.904 ] , [ 1.3 ] ] , dtype=np.float32 )

x_train = torch.from_numpy ( x_train )

y_train = torch.from_numpy ( y_train )


# 线性回归模型
class LinearRegression ( nn.Module ):
    def __init__(self):
        super ( LinearRegression , self ).__init__ ()
        # 建立层
        self.linear = nn.Linear ( 1 , 1 )  # 线性层，输入和输出都是一维

    def forward(self , x):
        out = self.linear ( x )
        return out


model = LinearRegression ()

# 定义loss和优化函数
criterion = nn.MSELoss ()
optimizer = torch.optim.SGD ( model.parameters () , lr=1e-4 )

# 开始训练
num_epochs = 1000
for epoch in range ( num_epochs ):
    inputs = x_train
    target = y_train

    # forward
    out = model ( inputs )
    loss = criterion ( out , target )
    # backward
    optimizer.zero_grad ()
    loss.backward ()
    optimizer.step ()

    if (epoch + 1) % 20 == 0:
        print ( f'Epoch[{epoch + 1}/{num_epochs}], loss: {loss.item ():.6f}' )

model.eval ()
with torch.no_grad ():
    predict = model ( x_train )
predict = predict.data.numpy ()

# 可视化
fig = plt.figure ( figsize=(10 , 5) )
plt.plot ( x_train.numpy () , y_train.numpy () , 'ro' , label='Original data' )
plt.plot ( x_train.numpy () , predict , label='Fitting Line' )
# 显示图例
plt.legend ()
plt.show ()

# 保存模型
torch.save ( model.state_dict () , './linear.pth' )
