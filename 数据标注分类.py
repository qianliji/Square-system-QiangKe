import os
import shutil
import dlib
import cv2
from PIL import Image
import numpy as np


def classify_images(source_folder):
    # 在源文件夹中创建子文件夹
    destination_folder_1024 = os.path.join(source_folder, "小于1024")
    destination_folder_400kb = os.path.join(source_folder, "低于400kb")

    # 创建子文件夹
    os.makedirs(destination_folder_1024, exist_ok=True)
    os.makedirs(destination_folder_400kb, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件是否是图片格式
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # 构造完整的文件路径
            file_path = os.path.join(source_folder, filename)
            # 获取文件大小（以字节为单位）
            file_size = os.path.getsize(file_path)
            # 判断文件大小是否低于400KB
            if file_size < 400 * 1024:
                # 移动文件到小于400kb文件夹
                shutil.move(file_path, os.path.join(destination_folder_400kb, filename))
                print("小于400kb图片分类完成！")
            else:
                # 使用with语句打开图片文件
                with Image.open(file_path) as img:
                    # 获取图片的宽度和高度
                    width, height = img.size
                    # 判断宽度和高度是否低于1024
                    if width < 1024 or height < 1024:
                        # 移动文件到1024文件夹
                        img.close()
                        shutil.move(file_path, os.path.join(destination_folder_1024, filename))
                        print("分辨率低于1024图片分类完成！")


def is_gray_map(img, threshold=15):
    """
    判断是否是灰度图

    Args:
    img: PIL读入的图像
    threshold: 判断阈值，图片3个通道间差的方差均值小于阈值则判断为灰度图

    Returns:
    bool值
    """
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False


def move_black_white_images(source_folder, destination_folder):
    """移动黑白图片到指定文件夹"""
    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            file_path = os.path.join(source_folder, filename)
            img = Image.open(file_path)
            if is_gray_map(img):
                img.close()
                shutil.move(file_path, os.path.join(destination_folder, filename))
                print(f"黑白图片 '{filename}' 已移动到目标文件夹")
            else:
                img.close()


def detect_and_move_faces(input_folder):
    """检测人脸并移动"""
    face_detection_folder = os.path.join(input_folder, "检测到人脸")
    os.makedirs(face_detection_folder, exist_ok=True)
    no_face_detection_folder = os.path.join(input_folder, "未检测到人脸")
    os.makedirs(no_face_detection_folder, exist_ok=True)

    # 初始化人脸检测器
    detector = dlib.get_frontal_face_detector()

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 使用dlib检测人脸
            faces = detector(gray)

            if len(faces) > 0:
                shutil.move(img_path, face_detection_folder)
                print(f"人脸已检测并移动到 {face_detection_folder}")
            else:
                shutil.move(img_path, no_face_detection_folder)
                print(f"未检测到人脸，图像已移动到 {no_face_detection_folder}")

    print("人脸检测图片处理完成！")


if __name__ == "__main__":
    # 获取当前文件夹路径
    print("开始执行小于400kb图片分类 和 分辨率低于1024图片分类")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    classify_images(current_folder)

    print("开始执行黑白图片分类")
    destination_folder = os.path.join(current_folder, "黑白图片")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    move_black_white_images(current_folder, destination_folder)

    print("开始执行人脸检测图片分类")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    detect_and_move_faces(current_folder)
