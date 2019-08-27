print(1 + 2)
print(7 / 5)
print(3 ** 2)

print(type(10)) # <class 'int'>
print(type(2.718)) # <class 'float'>
print(type("Hello")) # <class 'str'>

a = [1, 2, 3, 4, 5]
len(a)
print(a[0], a[4])
a[4] = 99
print(a[0:2]) # 获取索引为0到2（不包括2！）的元素
print(a[1:])  # 获取从索引为1的元素到最后一个元素
print(a[:3])  # 获取从第一个元素到索引为3（不包括3！）的元素
print(a[:-1]) # 获取从第一个元素到最后一个元素的前一个元素之间的元素
print(a[:-2]) # 获取从第一个元素到最后一个元素的前二个元素之间的元素

me = {'height':180} # 生成字典
print(me)
print(me['height'])

#--------------------------------------------------------#
class Man:
    # 构造函数, 只在生成类的实例时被调用一次。
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("David")
print(m.hello())
print(m.goodbye())

#--------------------------------------------------------#
import numpy as np
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x)) # <class 'numpy.ndarray'>

y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x / 2.0)

A = np.array([[1, 2], [3, 4]])
print(A.shape) # (2, 2)
print(A.dtype) # dtype('int64')
B = np.array([[3, 0],[0, 6]])
print(A + B) # array([[ 4, 2], [ 3, 10]])
print(A * B) # array([[ 3, 0], [ 0, 24]])
print(A * 10) # array([[ 10, 20], [ 30, 40]]), 标量10 被当作 2 × 2 的矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
# 一维数组B被“巧妙地”变成了和二维数组 A相同的形状, [[10, 20], [10, 20]], 然后再以对应元素的方式进行运算。
print(A * B) # array([[ 10, 40], [30, 80]])
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X[0])  # 第 0行
print(X[0][1]) # (0,1)的元素
for row in X:
    print(row)

X = X.flatten() # 将X转换为一维数组
print(X[np.array([0, 2, 4])]) # 获取索引为0、2、4的元素
print(X > 15) # array([ True, True, False, True, False, False], dtype=bool)
print(X[X>15]) # array([51, 55, 19])

#--------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.show()

y1 = np.sin(x)
y2 = np.cos(x)
# 绘制图形
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos") # 用虚线绘制
plt.xlabel("x") # x轴标签
plt.ylabel("y") # y轴标签
plt.title('sin & cos') # 标题
plt.legend()
plt.show()

from matplotlib.image import imread

img = imread('lena.png') # 读入图像（设定合适的路径！）
plt.imshow(img)
plt.show()