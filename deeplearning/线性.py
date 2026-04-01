import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_train = [1,2,3,4,5]
y_train = [3,5,7,9,11]
x_test = [7,8,9,10]
y_test = [15,17,19,21]

def forward(x):
    return w*x+b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2


w_list = []
b_list = []
mselist=[]
for w in np.arange(1.0,4.0,0.5):
    for b in np.arange(0,3.0,0.5):
        print(f'w是{w},b是{b}')
        l_sum = 0
        for x,y in zip(x_train,y_train):
            l_sum += loss(x,y)
        print(f'MSE是{l_sum/len(x_train)}')
        w_list.append(w)
        b_list.append(b)
        mselist.append(l_sum/len(x_train))  #代码拟合部分

w_unique = np.unique(w_list) #unique 是获取唯一值并进行排序
b_unique = np.unique(b_list)
W,B = np.meshgrid(w_unique,b_unique)#生成一一对应的二维数组
MSE_grid = np.array(mselist).reshape(len(b_unique),len(w_unique)).T #每一行对应一个b，每一列对应一个w
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W,B,MSE_grid,cmap = 'viridis',linewidth = 0 ,antialiased = True)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
ax.set_title('MSE损失函数')
plt.show()




