import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets , transforms

# 定义超参数
batch_size = 128
learning_rate = 1e-2
num_epoches = 20


def to_np(x):
    return x.cpu ().data.numpy ()


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST (
    root='./datasets' , train=True , transform=transforms.ToTensor () , download=True )

test_dataset = datasets.MNIST (
    root='./datasets' , train=False , transform=transforms.ToTensor () )

train_loader = DataLoader ( train_dataset , batch_size=batch_size , shuffle=True )
test_loader = DataLoader ( test_dataset , batch_size=batch_size , shuffle=False )


# 卷积神经网络模型-Convolutional Neural Network
class CNNet ( nn.Module ):
    def __init__(self , input_dim , n_classes):
        super ( CNNet , self ).__init__ ()
        self.conv = nn.Sequential (
            nn.Conv2d ( input_dim , 6 , kernel_size=3 , stride=1 , padding=1 ) ,  # 卷积层
            nn.ReLU ( True ) ,
            nn.MaxPool2d ( 2 , 2 ) ,
            nn.Conv2d ( 6 , 16 , kernel_size=5 , stride=1 , padding=0 ) ,
            nn.ReLU ( True ) ,
            nn.MaxPool2d ( 2 , 2 )
        )
        self.fc = nn.Sequential (
            nn.Linear ( 400 , 120 ) ,
            nn.Linear ( 120 , 84 ) ,
            nn.Linear ( 84 , n_classes )
        )

    def forward(self , x):
        out = self.conv ( x )
        out = out.view ( out.size ( 0 ) , -1 )
        out = self.fc ( out )
        return out


model = CNNet ( 1 , 10 )  # 图片大小是28x28
use_gpu = torch.cuda.is_available ()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda ()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss ()
optimizer = optim.SGD ( model.parameters () , lr=learning_rate )
logger = Logger ( './logs' )
# 开始训练
for epoch in range ( num_epoches ):
    print ( 'epoch {}'.format ( epoch + 1 ) )
    print ( '*' * 10 )
    running_loss = 0.0
    running_acc = 0.0
    for i , data in enumerate ( train_loader , 1 ):
        img , label = data
        if use_gpu:
            img = img.cuda ()
            label = label.cuda ()
        img = Variable ( img )
        label = Variable ( label )
        # 向前传播
        out = model ( img )
        loss = criterion ( out , label )
        running_loss += loss.data[ 0 ] * label.size ( 0 )
        _ , pred = torch.max ( out , 1 )
        num_correct = (pred == label).sum ()
        accuracy = (pred == label).float ().mean ()
        running_acc += num_correct.data[ 0 ]
        # 向后传播
        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()
        if i % 300 == 0:
            print ( '[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format (
                epoch + 1 , num_epoches , running_loss / (batch_size * i) ,
                running_acc / (batch_size * i) ) )
    print ( 'Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format (
        epoch + 1 , running_loss / (len ( train_dataset )) , running_acc / (len (
            train_dataset )) ) )
    model.eval ()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img , label = data
        if use_gpu:
            img = Variable ( img , volatile=True ).cuda ()
            label = Variable ( label , volatile=True ).cuda ()
        else:
            img = Variable ( img , volatile=True )
            label = Variable ( label , volatile=True )
        out = model ( img )
        loss = criterion ( out , label )
        eval_loss += loss.data[ 0 ] * label.size ( 0 )
        _ , pred = torch.max ( out , 1 )
        num_correct = (pred == label).sum ()
        eval_acc += num_correct.data[ 0 ]
    print ( 'Test Loss: {:.6f}, Acc: {:.6f}'.format ( eval_loss / (len (
        test_dataset )) , eval_acc / (len ( test_dataset )) ) )
    print ()

# 保存模型
torch.save ( model.state_dict () , './cnn.pth' )
