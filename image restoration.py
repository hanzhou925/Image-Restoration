###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
from matplotlib import pyplot as plt  # 展示图片
import numpy as np  # 数值处理
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析
# 图片路径
img_path = 'A.png'

# 以 BGR 方式读取图片
img = cv2.imread(img_path)

# 将 BGR 方式转换为 RGB 方式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 打印图片类型
print(type(img))

# 展示图片
plt.imshow(img)

# 关闭坐标轴
plt.axis('off')
def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path) 
    
    # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    return img
# 加载图片的路径和名称
img_path = 'A.png'

# 读取图片
img = read_image(img_path)  

# 读取图片后图片的类型
print(type(img))

# 展示图片
plt.imshow(img)  

# 关闭坐标轴
plt.axis('off') 

def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)
    
    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()
# 加载图片的路径和名称
img_path = 'A.png'

# 读取图片
img = read_image(img_path)

# 展示图片
plot_image(image=img, image_title="original image")
def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)
    
    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()
    
    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max
        
        # 转换图片数组数据类型
        img = img.astype(np.uint8)
    
    # 将 RGB 方式转换为 BGR 方式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)
# 图片路径和名称
img_path = 'A.png'

# 读取图片
img = read_image(img_path)

# 保存图片，保存成功则文件栏会出现 A_save_img.png
save_image(filename='A_{}_img.png'.format("save"), image=img)
def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型 
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)
    
    # 图像数组数据放缩在 0-1 之间
    return image.astype(np.double) / info.max
# 图片的路径和名称
img_path = 'A.png'

# 读取图片
img = read_image(img_path)

# 展示部分没有归一化的数据:
print("没有归一化的数据：\n", img[0, 0, :])

# 图片数据归一化
img = normalization(img)

# 展示部分 归一化后的数据
print("归一化后的数据：\n", img[0, 0, :])
print(img.shape)
print(img)

def noise_mask_image(img, noise_ratio=[0.8,0.4,0.6]):
    """
    根据题目要求生成受损图片
    :param img: cv2 读取图片,而且通道数顺序为 RGB
    :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
                        默认值分别是 [0.8,0.4,0.6]
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------
    row,col=img.shape[:2]
    rgb=[None,None,None] #rgb
    for i in range(3):
        #构建其中一个通道的噪声图
        for j in range(row):
            if rgb[i] is None:
                rgb[i]=np.random.choice(2,(1,col),p=[noise_ratio[i],1-noise_ratio[i]])
                #以对应比率生成噪声，值为0的概率即为噪声比率o, 值为1的概率对应（1-噪声比率），采样值为1*col
            else:
                a = np.random.choice(2,(1,col),p=[noise_ratio[i],1-noise_ratio[i]])
                
                rgb[i]=np.concatenate((rgb[i],a),axis=0) #数组拼接
    #扩展 shape
    for i in range(3):
        rgb[i]=rgb[i][:,:,np.newaxis]
        
    #合并生成噪声遮罩
    rst = np.concatenate((rgb[0],rgb[1],rgb[2]),axis=2)
    noise_img = rst*img#将噪声遮罩覆盖在原图上，即为受损图像
    # -----------------------------------------------
    return noise_img
noise_mask_image(img)
# 每个通道数不同的噪声比率
noise_ratio = [0.4, 0.6, 0.8]

# 图片路径及名称
img_path = 'A.png'

# 读取图片
img = read_image(img_path)

# 图片数据归一化
nor_img = normalization(img)

# 生成受损图片
noise_img = noise_mask_image(nor_img, noise_ratio)

# 判断还未生成受损图片时，则提示对方还未生成受损图片，否则展示受损图片
if noise_img is not None:
    # 展示受损图片
    # 图片名称
    image_title = "noise_mask_image"

    # 展示图片
    plot_image(noise_img, image_title)

else:
    print("返回值是 None, 请生成受损图片并返回!")
def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')
# 展示原始图片、受损图片、噪声图片。
# 原始图片路径
img_path = 'A.png'

# 读取图片
img = read_image(img_path)  

# 展示原始图片
plot_image(image=img, image_title="original image")

# 受损图片部分
# 图像数据归一化
nor_img = normalization(img) 

# 每个通道数不同的噪声比率
noise_ratio = [0.4, 0.6, 0.8]

# 生成受损图片
noise_img = noise_mask_image(nor_img, noise_ratio)

if noise_img is None:
    # 未生成受损图片
    print("返回值是 None, 请生成受损图片并返回!")
    
else:
    # 展示受损图片
    plot_image(image=noise_img, image_title="the noise_ratio = %s of original image"%noise_ratio)
    
    # 根据受损图片获取噪声图片
    noise_mask = get_noise_mask(noise_img) 
    
    # 展示噪声图片
    plot_image(image=noise_mask, image_title="the noise_ratio = %s of noise mask image"%noise_ratio)
def compute_error(res_img, img):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像 
    :param img:原始图像 
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    # 初始化
    error = 0.0
    
    # 将图像矩阵转换成为np.narray
    res_img = np.array(res_img)
    img = np.array(img)
    
    # 如果2个图像的形状不一致，则打印出错误结果，返回值为 None
    if res_img.shape != img.shape:
        print("shape error res_img.shape and img.shape %s != %s" % (res_img.shape, img.shape))
        return None
    
    # 计算图像矩阵之间的评估误差
    error = np.sqrt(np.sum(np.power(res_img - img, 2)))
    
    return round(error,3)
# 计算平面二维向量的 2-范数值 
img0 = [1, 0]
img1 = [0, 1]
print("平面向量的评估误差：", compute_error(img0, img1))
from skimage.measure import compare_ssim as ssim
from scipy import spatial

def calc_ssim(img, img_noise):
    """
    计算图片的结构相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    return ssim(img, img_noise,
                multichannel=True,
                data_range=img_noise.max() - img_noise.min())

def calc_csim(img, img_noise):
    """
    计算图片的 cos 相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    img = img.reshape(-1)
    img_noise = img_noise.reshape(-1)
    return 1 - spatial.distance.cosine(img, img_noise)
from PIL import Image
import numpy as np
def read_img(path):
    img = Image.open(path)
    img = img.resize((150,150))
    img = np.asarray(img, dtype="uint8")
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(img.dtype)
    # 图像数组数据放缩在 0-1 之间
    return img.astype(np.double) / info.max


img =  read_img('A.png')
noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1

img_noise = img + abs(noise)

print('相同图片的 SSIM 相似度: ', calc_ssim(img, img))
print('相同图片的 Cosine 相似度: ', calc_csim(img, img))
print('与噪声图片的 SSIM 相似度: ', calc_ssim(img, img_noise))
print('与噪声图片的 Cosine 相似度: ', calc_csim(img, img_noise))
def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------
    rows,cols, channel = res_img.shape
    region=10 #10*10
    row_cnt=rows//region
    col_cnt=cols//region
    #分割区域
    for chan in range(channel):
        for rn in range(row_cnt+1):
            ibase = rn * region
            if rn == row_cnt:#到边界返回
                ibase = rows - region
            for cn in range(col_cnt+1):
                jbase = cn*region
                if cn == col_cnt:#到边界返回
                    jbase = cols-region
                x_train=[]
                y_train=[]
                x_test=[]
                for i in range(ibase,ibase+region): #遍历每个10*10的区域
                    for j in range(jbase,jbase+region):
                        if noise_mask[i,j,chan]==0: #噪声点
                            x_test.append([i,j])#将噪声点加入测试集
                            continue
                        x_train.append([i,j]) #x训练集为坐标点
                        y_train.append([res_img[i,j,chan]]) #y训练集为对应坐标点像素值
                if x_train ==[]:
                    print("x_train is None")
                    continue
                reg = LinearRegression()
                reg.fit(x_train,y_train) #符合线性模型进行拟合
                pred = reg.predict(x_test)#对测试集进行预测
                
                for i in range(len(x_test)):
                    res_img[x_test[i][0],x_test[i][1],chan] = pred[i][0]
    
    res_img[res_img > 1.0]=1.0
    res_img[res_img<0.0]=0.0


    # ---------------------------------------------------------------

    return res_img
restore_image(noise_img, size=4)
# 原始图片
# 加载图片的路径和名称
img_path = 'A.png'

# 读取原始图片
img = read_image(img_path)

# 展示原始图片
plot_image(image=img, image_title="original image")

# 生成受损图片
# 图像数据归一化
nor_img = normalization(img)

# 每个通道数不同的噪声比率
noise_ratio = [0.4, 0.6, 0.8]

# 生成受损图片
noise_img = noise_mask_image(nor_img, noise_ratio)

if noise_img is not None:
    # 展示受损图片
    plot_image(image=noise_img, image_title="the noise_ratio = %s of original image"%noise_ratio)

    # 恢复图片
    res_img = restore_image(noise_img)
    
    # 计算恢复图片与原始图片的误差
    print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))
    print("恢复图片与原始图片的 SSIM 相似度: ", calc_ssim(res_img, nor_img))
    print("恢复图片与原始图片的 Cosine 相似度: ", calc_csim(res_img, nor_img))

    # 展示恢复图片
    plot_image(image=res_img, image_title="restore image")

    # 保存恢复图片
    save_image('res_' + img_path, res_img)
else:
    # 未生成受损图片
    print("返回值是 None, 请生成受损图片并返回!")