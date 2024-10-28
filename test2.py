import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm
import math

# 定义光场的行列数
num_rows, num_cols = 16, 16
image_height, image_width = 240, 320  # 每张图像的分辨率

# 初始化用于存储光场数据的 NumPy 数组
light_field_images = np.zeros((num_rows, num_cols, image_height, image_width, 3), dtype=np.uint8)
stuv_coordinates = np.zeros((num_rows, num_cols, image_height, image_width, 4), dtype=np.float64)  # 存储 s, t, u, v 坐标
main_camera_position = [0, 0]  # 中心相机位置，增加为可变数组
focal_length_factor = 1  # 焦距因子（模拟 z 轴）
focus_distance = 30  # 焦距参数

# 定义按钮的尺寸和位置
button_position = (100, image_height + 10)  # 按钮绘制在图像下方的区域
button_size = (100, 40)  # 按钮的宽度和高度

def load_image(filename, row_index, col_index):
    img = cv2.imread(filename)
    if img is not None:
        img = cv2.resize(img, (image_width, image_height))  # 确保图像大小一致
        light_field_images[row_index, col_index] = img  # 将图像存储在光场数组中
    else:
        print(f"图像 {filename} 未找到")

def load_light_field_images_parallel(image_folder):
    counter = 1
    tasks = []
    with ThreadPoolExecutor() as executor:
        for row in range(num_rows):
            for col in range(num_cols):
                filename = os.path.join(image_folder, f'lowtoys{counter:03d}.bmp')
                tasks.append(executor.submit(load_image, filename, row, col))
                counter += 1
    # 确保所有图像读取完成
    for task in tasks:
        task.result()

def render_final_image():
    final_image = np.zeros_like(light_field_images[0, 0], dtype=np.float64)  # 初始化合成图像
    height, width = final_image.shape[:2]

    y_pos = main_camera_position[1] * (num_rows - 1)
    x_pos = main_camera_position[0] * (num_cols - 1)
    valid_camera_views = []  # 存储有效的相机视角
    weights = []  # 存储每个相机的权重

    for row in range(num_rows):
        for col in range(num_cols):
            distance = math.sqrt((x_pos - col) ** 2 + (y_pos - row) ** 2)  # 计算距离
            weight = norm.pdf(distance, loc=0.0, scale=1.0)  # 计算正态分布的权重
            if weight > 0.0001:  # 忽略小权重
                dx = int((col - x_pos) * focal_length_factor * width / focus_distance)  # x方向位移
                dy = int(-(row - y_pos) * focal_length_factor * height / focus_distance)  # y方向位移
                valid_camera_views.append((row, col, dx, dy))  # 存储相机索引和位移
                weights.append(weight)  # 存储权重

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    for i in range(height):
        for j in range(width):
            for idx, (row, col, dx, dy) in enumerate(valid_camera_views):
                if 0 <= i + dy < height and 0 <= j + dx < width:  # 确保像素位置有效
                    final_image[i, j] += normalized_weights[idx] * light_field_images[row, col][i + dy, j + dx]

    # 模拟 z 轴拉远时的模糊效果
    blur_kernel_size = max(1, int(focal_length_factor * 2))  # 根据 z 轴调整模糊程度
    if blur_kernel_size % 2 == 0:  # 保证卷积核为奇数
        blur_kernel_size += 1
    final_image = cv2.GaussianBlur(final_image, (blur_kernel_size, blur_kernel_size), 0)

    # 将 float64 转换回 uint8 以便显示
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    return final_image

def on_focus_distance_change(val):
    global focus_distance
    focus_distance = max(val, 1)  # 避免焦距为0

def on_camera_position_x_change(val):
    global main_camera_position
    main_camera_position[0] = 1 - val / 100.0  # x轴位置反向调整

def on_camera_position_y_change(val):
    global main_camera_position
    main_camera_position[1] = 1 - val / 100.0  # y轴位置反向调整

def on_focal_length_factor_change(val):
    global focal_length_factor
    focal_length_factor = val / 10.0  # 焦距因子用于模拟 z 轴，范围由滑动条决定

def update_display():
    # 生成新的图像
    final_image = render_final_image()

    # 扩展图像高度以包含按钮区域
    extended_image = np.zeros((image_height + 60, image_width, 3), dtype=np.uint8)
    extended_image[0:image_height, :, :] = final_image  # 将合成图像放在扩展区域的顶部

    # 在扩展区域绘制按钮
    cv2.rectangle(extended_image, button_position, 
                  (button_position[0] + button_size[0], button_position[1] + button_size[1]), 
                  (0, 255, 0), -1)  # 绿色的按钮
    cv2.putText(extended_image, 'Confirm', 
                (button_position[0] + 10, button_position[1] + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 显示扩展后的图像
    cv2.imshow("HomeWork1", extended_image)

def handle_mouse_click(event, x, y, flags, param):
    # 检测按钮是否被点击
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_position[0] <= x <= button_position[0] + button_size[0] and \
           button_position[1] <= y <= button_position[1] + button_size[1]:
            print("确定按钮被点击")
            update_display()  # 点击按钮后更新显示

def main():
    image_folder = './toyLF'  # 图片存放的路径
    load_light_field_images_parallel(image_folder)

    cv2.namedWindow("HomeWork1", cv2.WINDOW_AUTOSIZE)

    # 创建滑动条控制焦距参数
    cv2.createTrackbar('focus_distance', 'HomeWork1', int(focus_distance), 100, on_focus_distance_change)

    # 创建滑动条控制相机位置
    cv2.createTrackbar('camera_x', 'HomeWork1', int((1 - main_camera_position[0]) * 100), 100, on_camera_position_x_change)
    cv2.createTrackbar('camera_y', 'HomeWork1', int((1 - main_camera_position[1]) * 100), 100, on_camera_position_y_change)

    # 创建滑动条控制 z 轴（视角深度）
    cv2.createTrackbar('camera_z', 'HomeWork1', int(focal_length_factor * 10), 100, on_focal_length_factor_change)

    # 设置鼠标回调函数，监听点击操作
    cv2.setMouseCallback("HomeWork1", handle_mouse_click)

    # 显示初始图像
    update_display()

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
