import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义光场的行列数
rows, cols = 8, 32
image_height, image_width = 240, 320  # 每张图像的分辨率

# 初始化用于存储光场数据的 NumPy 数组
light_field = np.zeros((rows, cols, image_height, image_width, 3), dtype=np.uint8)

# 读取图像数据
counter = 1  # 文件名计数器从001开始
for i in range(rows):
    for j in range(cols):
        # 生成文件名：lowtoys001.bmp 到 lowtoys256.bmp
        filename = fr'C:\Users\Zhaoyu Jimmy Qiu\Desktop\CS276\toyLF\lowtoys{counter:03d}.bmp'
        img = cv2.imread(filename)
        if img is not None:
            img = cv2.resize(img, (image_width, image_height))  # 确保图像大小一致
            light_field[i, j] = img  # 将图像存储在正确的行列位置
        else:
            print(f"图像 {filename} 未找到")
        counter += 1  # 每读取一张图像，计数器加1


print(light_field.shape)

# 显示读取的第0行第0列图像
plt.imshow(cv2.cvtColor(light_field[1, 0], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
