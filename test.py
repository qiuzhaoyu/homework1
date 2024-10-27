import cv2
import numpy as np
import math

# 定义相机网格的大小
camera_H = 16  # 高度（相机垂直方向上的数量）
camera_W = 16  # 宽度（相机水平方向上的数量）
camera_Tot = camera_W * camera_H  # 总的相机数量
image_Name = [""] * (camera_Tot + 10)  # 用于存储所有图像文件路径的列表
image = [None] * (camera_Tot + 10)  # 用于存储所有图像的矩阵
camera_Main_Pos = (0.52, 0.52)  # 主相机位置（位于网格中央）
image_Final = None  # 最终合成的图像
f = 1  # 焦距因子
z1 = 30  # 初始焦距参数1
z2 = 35  # 初始焦距参数2

def read_Image():
    """
    读取所有相机视角对应的图像文件，并存储在 'image' 列表中。
    使用 OpenCV 读取图片，如果文件不存在或无法读取，则返回 False。
    """
    global image
    for i in range(camera_Tot):  # 遍历所有相机视角
        # 根据文件编号格式生成文件路径名称
        if i < 9:
            image_Name[i] = f"./toyLF/lowtoys00{i+1}.bmp"
        elif i < 99:
            image_Name[i] = f"./toyLF/lowtoys0{i+1}.bmp"
        else:
            image_Name[i] = f"./toyLF/lowtoys{i+1}.bmp"

    # 使用 OpenCV 逐个读取每个图像文件
    for i in range(camera_Tot):
        readimg = cv2.imread(image_Name[i])  # 读取图像
        if readimg is None:  # 如果读取失败，输出错误信息
            print(f"Could not open or find the image: {image_Name[i]}")
            return False
        image[i] = readimg.astype(np.float64) / 255.0  # 将图像像素归一化至 [0, 1] 范围内
    return True

def naive_Image():
    """
    使用双线性插值的方式计算并生成最终的合成图像。
    """
    global image_Final
    image_Final = np.zeros_like(image[0])  # 初始化合成图像，大小与单个输入图像相同
    height, width = image_Final.shape[:2]  # 获取图像的高度和宽度

    # 计算主相机位置在网格中的浮点坐标
    y = 1 + camera_Main_Pos[1] * (camera_H - 1)
    x = 1 + camera_Main_Pos[0] * (camera_W - 1)
    x1, x2 = math.floor(x), math.ceil(x)  # 获取 x 方向上的相邻整数坐标
    y1, y2 = math.floor(y), math.ceil(y)  # 获取 y 方向上的相邻整数坐标

    # 对每个像素进行插值计算
    for i in range(height):
        for j in range(width):
            if x1 != x2:
                # x方向上的插值
                a = abs(x2 - x) * image[y1 * camera_W + x1][i, j] + abs(x1 - x) * image[y1 * camera_W + x2][i, j]
            else:
                a = image[y1 * camera_W + x1][i, j]

            if y1 != y2:
                # y方向上的插值
                b = abs(y2 - y) * image[y2 * camera_W + x1][i, j] + abs(y1 - y) * image[y2 * camera_W + x2][i, j]
                image_Final[i, j] = abs(y2 - y) * a + abs(y1 - y) * b
            else:
                image_Final[i, j] = a

def get_Normal_pdf(x, m, s):
    """
    计算正态分布概率密度函数，用于距离权重的计算。
    x: 距离
    m: 均值
    s: 标准差
    返回值为正态分布的概率密度。
    """
    inv_sqrt_2pi = 0.3989422804014327  # 1 / sqrt(2*pi)
    a = (x - m) / s  # 计算标准化值
    return inv_sqrt_2pi / s * np.exp(-0.5 * a * a)  # 返回正态分布的概率密度值

def advance_Image():
    """
    使用基于高斯加权插值的方式生成最终合成图像。
    """
    global image_Final
    image_Final = np.zeros_like(image[0])  # 初始化输出图像
    height, width = image_Final.shape[:2]  # 获取图像尺寸

    # 计算主相机位置在网格中的浮点坐标
    y = 1 + camera_Main_Pos[1] * (camera_H - 1)
    x = 1 + camera_Main_Pos[0] * (camera_W - 1)
    points = []  # 存储有效的相机视角
    weights = []  # 存储每个相机的权重

    # 计算每个相机视角的权重
    for a in range(1, camera_W + 1):
        for b in range(1, camera_H + 1):
            num = (b - 1) * camera_W + a  # 当前相机的编号
            dis = math.sqrt((x - a) ** 2 + (y - b) ** 2)  # 计算距离
            w = get_Normal_pdf(dis, 0.0, 1.0)  # 计算正态分布的权重
            if w > 0.0001:  # 忽略小权重
                dx = int((a - x) * f * width / z2)  # x方向上的位移
                dy = int(-(b - y) * f * height / z1)  # y方向上的位移
                points.append((num, dx, dy))  # 存储相机索引和位移
                weights.append(w)  # 存储权重

    # 对每个像素进行插值计算
    for i in range(height):
        for j in range(width):
            total_weight = sum(weights)  # 总权重
            if total_weight < 1e-8:  # 如果总权重太小，跳过该像素
                continue

            # 归一化权重
            normalized_weights = [w / total_weight for w in weights]
            for idx, (num, dx, dy) in enumerate(points):
                if 0 <= i + dy < height and 0 <= j + dx < width:  # 确保像素位置有效
                    image_Final[i, j] += normalized_weights[idx] * image[num][i + dy, j + dx]  # 插值计算

def main():
    """
    主函数，控制图像渲染和交互流程。
    - 读取图像
    - 生成插值图像
    - 通过键盘调整焦距并更新显示
    """
    if not read_Image():  # 读取光场图像
        return

    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)  # 创建显示窗口
    advance_Image()  # 生成初始插值图像
    cv2.imshow("Display window", image_Final)  # 显示初始插值图像
    
    dz = 0.1  # 焦距变化步长
    while True:
        key = cv2.waitKey(10)  # 等待键盘输入
        if key == ord('q'):  # 按 'q' 键退出
            break
        elif key == 0:
            global z2
            z2 += dz  # 增加 z2
        elif key == 1:
            z2 -= dz  # 减小 z2
        elif key == 2:
            global z1
            z1 -= dz  # 减小 z1
        elif key == 3:
            z1 += dz  # 增加 z1

        if z1 == 0 or z2 == 0:
            break

        if key in [0, 1, 2, 3]:  # 如果有焦距调整
            print(f"z1: {z1:.3f}, z2: {z2:.3f}")
            advance_Image()  # 重新生成插值图像
            cv2.imshow("Display window", image_Final)  # 显示更新后的图像

    cv2.waitKey(0)  # 等待用户按键

if __name__ == "__main__":
    main()
