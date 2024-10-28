import os
import cv2
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm

# 定义光场的行列数
num_rows, num_cols = 16, 16
image_height, image_width = 240, 320  # 每张图像的分辨率

# 初始化用于存储光场数据的 NumPy 数组
light_field_images = np.zeros((num_rows, num_cols, image_height, image_width, 3), dtype=np.uint8)
main_camera_position = [0.5, 0.5]  # 中心相机位置
focal_length_factor = 1  # 焦距因子（模拟 z 轴）
focus_distance = 30  # 焦距参数
aperture_size = 1  # 光圈大小
interpolation_method = 0  # 0: 双线性，1: 四线性

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

def bilinear_interpolate(img, x, y):
    """双线性插值方法"""
    x0 = int(x)
    x1 = min(x0 + 1, img.shape[1] - 1)
    y0 = int(y)
    y1 = min(y0 + 1, img.shape[0] - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def trilinear_interpolate(cube, x, y):
    """四线性插值方法"""
    x0 = max(0, int(x))
    x1 = min(x0 + 1, cube.shape[2] - 1)
    y0 = max(0, int(y))
    y1 = min(y0 + 1, cube.shape[1] - 1)

    if y1 >= cube.shape[0]:
        y1 = cube.shape[0] - 1
    if x1 >= cube.shape[2]:
        x1 = cube.shape[2] - 1

    c00 = cube[y0, x0]  # 左上
    c01 = cube[y0, x1]  # 右上
    c10 = cube[y1, x0]  # 左下
    c11 = cube[y1, x1]  # 右下

    xd = x - x0
    yd = y - y0

    top = c00 * (1 - xd) + c01 * xd  # 上面插值
    bottom = c10 * (1 - xd) + c11 * xd  # 下面插值

    return top * (1 - yd) + bottom * yd  # 最终插值

def render_final_image():
    final_image = np.zeros((image_height, image_width, 3), dtype=np.float64)  # 初始化合成图像
    height, width = final_image.shape[:2]

    y_pos = main_camera_position[1] * (num_rows - 1)
    x_pos = main_camera_position[0] * (num_cols - 1)
    valid_camera_views = []  # 存储有效的相机视角
    weights = []  # 存储每个相机的权重

    for row in range(num_rows):
        for col in range(num_cols):
            distance = math.sqrt((x_pos - col) ** 2 + (y_pos - row) ** 2)  # 计算距离
            weight = norm.pdf(distance, loc=0.0, scale=aperture_size)  # 使用光圈大小调整权重
            if weight > 0.0001:  # 忽略小权重
                dx = (col - x_pos) * focal_length_factor * width / focus_distance  # x方向位移
                dy = -(row - y_pos) * focal_length_factor * height / focus_distance  # y方向位移
                valid_camera_views.append((row, col, dx, dy))  # 存储相机索引和位移
                weights.append(weight)  # 存储权重

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    for i in range(height):
        for j in range(width):
            pixel_value = np.zeros(3, dtype=np.float64)
            for idx, (row, col, dx, dy) in enumerate(valid_camera_views):
                x = j + dx
                y = i + dy
                if 0 <= x < image_width and 0 <= y < image_height:  # 确保像素位置有效
                    if interpolation_method == 0:
                        interpolated_value = bilinear_interpolate(light_field_images[row, col], x, y)
                    else:
                        interpolated_value = trilinear_interpolate(light_field_images[row, col], x, y)
                    
                    pixel_value += normalized_weights[idx] * interpolated_value  # 确保插值结果是 RGB 值

            final_image[i, j] = pixel_value

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

def on_aperture_size_change(val):
    global aperture_size
    aperture_size = max(val, 1)  # 避免光圈大小为0，设置最小值为1

def on_interpolation_method_change(val):
    global interpolation_method
    interpolation_method = val  # 更新插值方法

def update_display():
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
    cv2.createTrackbar('f_distance', 'HomeWork1', int(focus_distance), 100, on_focus_distance_change)

    # 创建滑动条控制相机位置
    cv2.createTrackbar('camera_x', 'HomeWork1', int((1 - main_camera_position[0]) * 100), 100, on_camera_position_x_change)
    cv2.createTrackbar('camera_y', 'HomeWork1', int((1 - main_camera_position[1]) * 100), 100, on_camera_position_y_change)

    # 创建滑动条控制 z 轴（视角深度）
    cv2.createTrackbar('camera_z', 'HomeWork1', int(focal_length_factor * 10), 100, on_focal_length_factor_change)

    # 创建滑动条控制光圈大小
    cv2.createTrackbar('aperture', 'HomeWork1', int(aperture_size), 30, on_aperture_size_change)

    # 创建滑动条选择插值方法
    cv2.createTrackbar('interp', 'HomeWork1', interpolation_method, 1, on_interpolation_method_change)

    # 设置鼠标回调函数，监听点击操作
    cv2.setMouseCallback("HomeWork1", handle_mouse_click)

    # 显示初始图像
    update_display()

    cv2.waitKey(0)

if __name__ == "__main__":
    main()