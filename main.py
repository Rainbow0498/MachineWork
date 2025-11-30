import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QMessageBox,
    QStackedWidget
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage


# --------- 工具函数：中文路径安全读写 ---------
def cv_imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """兼容中文路径的 imread"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


def cv_imwrite_unicode(path, img):
    """兼容中文路径的 imwrite"""
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


# --------- 工具函数：cv2 图像转 QImage ---------
def cv2_to_qimage(cv_img):
    """
    将 OpenCV 的图像(numpy 数组)转换为 QImage，支持 BGR 彩色和单通道灰度。
    """
    if cv_img is None:
        return QImage()

    if len(cv_img.shape) == 2:
        # 灰度图
        h, w = cv_img.shape
        bytes_per_line = w
        return QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
    else:
        # 彩色 BGR -> RGB
        cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_rgb.shape
        bytes_per_line = ch * w
        return QImage(cv_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


# --------- 自定义算法实现部分 ---------
def custom_rotate_90_clockwise(img):

    if img.ndim == 2:
        # 灰度图：先转置，再上下翻转
        return np.flipud(img.T)
    else:
        # 彩色图：交换 H,W 轴，再上下翻转
        return np.flipud(np.transpose(img, (1, 0, 2)))


def custom_invert(img):
    # 假定 uint8
    return 255 - img


def custom_to_gray(img):
    """自定义：灰度化。手写 BGR -> Gray"""
    if img.ndim == 2:
        # 本来就是灰度
        return img.copy()

    # BGR 顺序
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)

    gray = 0.114 * b + 0.587 * g + 0.299 * r
    gray = np.clip(gray, 0, 255)
    return gray.astype(np.uint8)


def custom_resize_nearest(img, scale):
    """自定义：最近邻缩放（不调用 cv2.resize）"""
    if scale <= 0:
        raise ValueError("scale must be > 0")

    h, w = img.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # 映射关系：目标像素 (y, x) 对应原图像素 (y/scale, x/scale)
    y_idx = (np.arange(new_h) / scale).astype(int)
    x_idx = (np.arange(new_w) / scale).astype(int)
    y_idx = np.clip(y_idx, 0, h - 1)
    x_idx = np.clip(x_idx, 0, w - 1)

    if img.ndim == 2:
        out = img[y_idx[:, None], x_idx[None, :]]
    else:
        out = img[y_idx[:, None], x_idx[None, :], :]

    return out


# --------- 缩略图控件 ---------
class ThumbnailLabel(QLabel):
    clicked = pyqtSignal(str)  # 传出图片路径

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)
        super().mousePressEvent(event)


# --------- 主窗口 ---------
class ImageBrowserMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("算法对比图片浏览器")
        self.resize(1400, 800)

        # 状态变量
        self.image_paths = []          # 浏览器中所有图片路径
        self.current_image_path = None
        self.original_image = None     # 原始图像

        self.custom_image = None       # 自定义算法版
        self.cv_image = None           # OpenCV 算法版

        self._init_ui()

    # ---------- UI 初始化 ----------
    def _init_ui(self):
        # 0 = 浏览器页面, 1 = 编辑页面
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # ===== 浏览器页面 =====
        browser_page = QWidget()
        browser_layout = QVBoxLayout(browser_page)

        # 顶部按钮
        top_btn_layout = QHBoxLayout()
        self.btn_open_files = QPushButton("打开图片（多选）")
        self.btn_open_dir = QPushButton("打开目录")
        self.btn_open_files.clicked.connect(self.open_images)
        self.btn_open_dir.clicked.connect(self.open_directory)
        top_btn_layout.addWidget(self.btn_open_files)
        top_btn_layout.addWidget(self.btn_open_dir)
        top_btn_layout.addStretch()
        browser_layout.addLayout(top_btn_layout)

        # 缩略图区域（滚动 + 网格）
        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)

        self.thumb_container = QWidget()
        self.thumb_layout = QGridLayout(self.thumb_container)
        self.thumb_layout.setContentsMargins(10, 10, 10, 10)
        self.thumb_layout.setSpacing(10)

        self.thumb_scroll.setWidget(self.thumb_container)
        browser_layout.addWidget(self.thumb_scroll)

        self.stack.addWidget(browser_page)

        # ===== 编辑页面（算法对比）=====
        editor_page = QWidget()
        editor_layout = QHBoxLayout(editor_page)

        # 左侧：两个对比图
        images_layout = QHBoxLayout()

        # 自定义算法区域
        custom_area = QVBoxLayout()
        self.custom_title = QLabel("自定义算法结果")
        self.custom_title.setAlignment(Qt.AlignCenter)
        self.custom_title.setStyleSheet("color: black; font-weight: bold;")

        self.custom_label = QLabel()
        self.custom_label.setAlignment(Qt.AlignCenter)
        self.custom_label.setStyleSheet("background-color: #202020;")

        custom_scroll = QScrollArea()
        custom_scroll.setWidgetResizable(True)
        custom_scroll.setWidget(self.custom_label)

        custom_area.addWidget(self.custom_title)
        custom_area.addWidget(custom_scroll)

        # OpenCV 算法区域
        cv_area = QVBoxLayout()
        self.cv_title = QLabel("OpenCV 算法结果")
        self.cv_title.setAlignment(Qt.AlignCenter)
        self.cv_title.setStyleSheet("color: black; font-weight: bold;")

        self.cv_label = QLabel()
        self.cv_label.setAlignment(Qt.AlignCenter)
        self.cv_label.setStyleSheet("background-color: #202020;")

        cv_scroll = QScrollArea()
        cv_scroll.setWidgetResizable(True)
        cv_scroll.setWidget(self.cv_label)

        cv_area.addWidget(self.cv_title)
        cv_area.addWidget(cv_scroll)

        images_layout.addLayout(custom_area, stretch=1)
        images_layout.addLayout(cv_area, stretch=1)

        editor_layout.addLayout(images_layout, stretch=3)

        # 右侧：操作按钮
        right_panel = QVBoxLayout()

        self.btn_rotate = QPushButton("旋转90°")
        self.btn_zoom_in = QPushButton("放大")
        self.btn_zoom_out = QPushButton("缩小")
        self.btn_invert = QPushButton("反色")
        self.btn_gray = QPushButton("灰度化")
        self.btn_reset = QPushButton("重置")
        self.btn_save = QPushButton("保存（自定义算法结果）")
        self.btn_back = QPushButton("返回浏览器")

        self.btn_rotate.clicked.connect(self.rotate_90)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_invert.clicked.connect(self.invert_colors)
        self.btn_gray.clicked.connect(self.to_gray)
        self.btn_reset.clicked.connect(self.reset_image)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_back.clicked.connect(self.back_to_browser)

        for b in [
            self.btn_rotate, self.btn_zoom_in, self.btn_zoom_out,
            self.btn_invert, self.btn_gray, self.btn_reset,
            self.btn_save, self.btn_back
        ]:
            b.setMinimumHeight(40)
            right_panel.addWidget(b)

        right_panel.addStretch()
        editor_layout.addLayout(right_panel, stretch=1)

        self.stack.addWidget(editor_page)

    # ---------- 浏览器：加载图片 ----------
    def open_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff)"
        )
        if not files:
            return

        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)

        self.refresh_thumbnails()
        self.stack.setCurrentIndex(0)

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择图片目录", "")
        if not directory:
            return

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
        for root, dirs, files in os.walk(directory):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    path = os.path.join(root, name)
                    if path not in self.image_paths:
                        self.image_paths.append(path)

        self.image_paths.sort()
        self.refresh_thumbnails()
        self.stack.setCurrentIndex(0)

    # ---------- 缩略图 ----------
    def refresh_thumbnails(self):
        # 清空旧缩略图
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        thumb_size = 150
        cols = 5
        row = 0
        col = 0

        for img_path in self.image_paths:
            label = ThumbnailLabel(img_path)
            pix = QPixmap(img_path)
            if pix.isNull():
                continue
            pix = pix.scaled(thumb_size, thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pix)
            label.setFixedSize(thumb_size, thumb_size)
            label.setStyleSheet("border:1px solid #555;")

            label.clicked.connect(self.open_image_in_editor)

            self.thumb_layout.addWidget(label, row, col)

            col += 1
            if col >= cols:
                col = 0
                row += 1

    # ---------- 打开图片进入对比界面 ----------
    def open_image_in_editor(self, image_path):
        self.load_image(image_path)
        self.stack.setCurrentIndex(1)

    def load_image(self, image_path):
        print("准备加载：", image_path)
        print("exists:", os.path.exists(image_path))

        img = cv_imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法加载图片：{image_path}")
            return

        self.current_image_path = image_path
        self.original_image = img.copy()

        # 初始时，两侧都是原图
        self.custom_image = self.original_image.copy()
        self.cv_image = self.original_image.copy()

        self.update_image_labels()

    # ---------- 更新对比图显示 ----------
    def update_image_labels(self):
        if self.custom_image is not None:
            qimg_custom = cv2_to_qimage(self.custom_image)
            pix_custom = QPixmap.fromImage(qimg_custom)
            self.custom_label.setPixmap(pix_custom)
            self.custom_label.adjustSize()

        if self.cv_image is not None:
            qimg_cv = cv2_to_qimage(self.cv_image)
            pix_cv = QPixmap.fromImage(qimg_cv)
            self.cv_label.setPixmap(pix_cv)
            self.cv_label.adjustSize()

    # ---------- 图像操作：每次对 custom 和 cv 两套图分别处理 ----------
    def rotate_90(self):
        if self.custom_image is None or self.cv_image is None:
            return

        # 自定义算法版本
        self.custom_image = custom_rotate_90_clockwise(self.custom_image)

        # OpenCV 内置版本
        self.cv_image = cv2.rotate(self.cv_image, cv2.ROTATE_90_CLOCKWISE)

        self.update_image_labels()

    def zoom_in(self):
        if self.custom_image is None or self.cv_image is None:
            return
        scale = 1.25

        # 自定义最近邻缩放
        self.custom_image = custom_resize_nearest(self.custom_image, scale)

        # OpenCV 缩放（双线性插值）
        h, w = self.cv_image.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        self.cv_image = cv2.resize(self.cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.update_image_labels()

    def zoom_out(self):
        if self.custom_image is None or self.cv_image is None:
            return
        scale = 1 / 1.25

        # 自定义最近邻缩小
        self.custom_image = custom_resize_nearest(self.custom_image, scale)

        # OpenCV 缩小
        h, w = self.cv_image.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        self.cv_image = cv2.resize(self.cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.update_image_labels()

    def invert_colors(self):
        if self.custom_image is None or self.cv_image is None:
            return

        # 自定义反色
        self.custom_image = custom_invert(self.custom_image)

        # OpenCV 反色
        self.cv_image = cv2.bitwise_not(self.cv_image)

        self.update_image_labels()

    def to_gray(self):
        if self.custom_image is None or self.cv_image is None:
            return

        # 自定义灰度
        self.custom_image = custom_to_gray(self.custom_image)

        # OpenCV 灰度
        if len(self.cv_image.shape) == 3:
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        self.update_image_labels()

    def reset_image(self):
        if self.original_image is None:
            return

        self.custom_image = self.original_image.copy()
        self.cv_image = self.original_image.copy()
        self.update_image_labels()

    # ---------- 保存 + 返回 ----------
    def save_image(self):
        if self.custom_image is None:
            return

        default_dir = os.path.dirname(self.current_image_path) if self.current_image_path else ""
        default_name = "custom_edited_" + os.path.basename(self.current_image_path) if self.current_image_path else "custom_edited.png"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片（自定义算法结果）",
            os.path.join(default_dir, default_name),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"
        )
        if not save_path:
            return

        success = cv_imwrite_unicode(save_path, self.custom_image)
        if not success:
            QMessageBox.warning(self, "错误", "保存失败！")
            return

        if save_path not in self.image_paths:
            self.image_paths.append(save_path)
            self.image_paths.sort()
            self.refresh_thumbnails()

        QMessageBox.information(self, "提示", f"已保存自定义算法结果到：\n{save_path}")

    def back_to_browser(self):
        self.stack.setCurrentIndex(0)


# --------- 程序入口 ---------
def main():
    app = QApplication(sys.argv)
    win = ImageBrowserMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
