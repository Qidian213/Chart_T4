import numpy as np
import random
import cv2

class ChartAugmenter(object):
    """
    图像增强
    """

    @staticmethod
    def augment(src_img):
        ls_fun_ = [ChartAugmenter.invert_color, ChartAugmenter.jpeg, ChartAugmenter.gaussian_blur,
                   ChartAugmenter.mean_blur, ChartAugmenter.add_salt_and_pepper_noise, ChartAugmenter.add_speckle_noise,
                   ChartAugmenter.reshape_blur, ChartAugmenter.random_color_distort]
        num = len(ls_fun_)
        choice = random.randint(0, num - 1)
        return ls_fun_[choice](src_img).astype(np.uint8)

    @staticmethod
    def invert_color(image_array: np.ndarray):
        """
        反色处理
        :param image_array:
        :return:
        """
        return 255 - image_array

    @staticmethod
    def gaussian_blur(image_array: np.ndarray):
        """
        高斯模糊
        :param image_array:
        :return:
        """
        var = np.random.randint(0, 1)
        kernel = (var + 1) * 2 + 1
        return cv2.GaussianBlur(image_array, (kernel, kernel), var)

    @staticmethod
    def mean_blur(image_array: np.ndarray):
        """
        均值模糊
        :param image_array:
        :return:
        """
        kernel = 3
        return cv2.blur(image_array, (kernel, kernel))

    @staticmethod
    def add_salt_and_pepper_noise(image_array: np.ndarray, noise_prob: float=0.01, salt_vs_pepper: float=0.5):
        """
        # 椒盐噪声
        :param noise_prob:
        :param salt_vs_pepper:
        :param image_array:
        :return:
        """
        salt_ratio = 1 / (1 + salt_vs_pepper)
        pepper_ratio = 1 - salt_ratio
        h, w, c = image_array.shape
        out = np.copy(image_array)

        # salt mode
        num_salt = int(np.ceil(h * w * noise_prob * salt_ratio))
        coords = tuple([np.random.randint(0, size - 1, num_salt) for size in (h, w)])
        out[coords] = 255

        # pepper mode
        num_pepper = int(np.ceil(h * w * noise_prob * pepper_ratio))
        coords = tuple([np.random.randint(0, size - 1, num_pepper) for size in (h, w)])
        out[coords] = 0

        return out

    @staticmethod
    def reshape_blur(image_array: np.ndarray):
        """
        对图片先进行下采样，再上采样回原来的尺寸
        :param image_array:
        :return:
        """
        w, h = image_array.shape[:2]
        image_array = cv2.resize(image_array, (h // 2, w // 2), interpolation=cv2.INTER_AREA)
        image_array = cv2.resize(image_array, (h, w), interpolation=cv2.INTER_LINEAR)
        return image_array

    @staticmethod
    def jpeg(image_array):
        """
        jpeg压缩
        :param image_array:
        :return:
        """
        quality = np.random.randint(50, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, enc_img = cv2.imencode('.jpg', image_array, encode_param)
        dec_img = cv2.imdecode(enc_img, 1)
        return dec_img

    @staticmethod
    def grey(image_array):
        """
        转为灰度图
        :param image_array:
        :return:
        """
        grey_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        grey_img = cv2.cvtColor(grey_img, cv2.COLOR_GRAY2RGB)
        return grey_img

    @staticmethod
    def binary(image_array):
        """
        二值化图片
        :param image_array:
        :return:
        """
        bg_index = np.where((image_array[:, :, 0] == 255) &
                            (image_array[:, :, 1] == 255) &
                            (image_array[:, :, 2] == 255))
        if np.random.random() < .8:
            bg_color = np.random.randint(170, 255)
            front_color = np.random.randint(10, 120)
        else:
            bg_color = np.random.randint(10, 120)
            front_color = np.random.randint(170, 255)
        image_array[:, :, :] = front_color
        image_array[bg_index] = bg_color
        return image_array

    @staticmethod
    def add_speckle_noise(image_array):
        """
         用方程g=f + n*f将乘性噪声添加到图像f上
        :param image_array:
        :return:
        """
        noise = np.random.normal(0, 0.04, image_array.shape)
        image_array = image_array + noise * image_array
        return np.clip(image_array, 0, 255).astype('uint8')

    @staticmethod
    def vertical_flip(image_array):
        """
        竖直翻转
        :param image_array:
        :return:
        """
        return np.flipud(image_array)

    @staticmethod
    def horizontal_flip(image_array):
        """
        水平翻转
        :param image_array:
        :return:
        """
        return np.fliplr(image_array)

    @staticmethod
    def random_color_distort(image_array, brightness_delta=32, contrast_low=0.6, contrast_high=1.4,
                             saturation_low=0.6, saturation_high=1.4, hue_delta=102):
        """图片颜色畸变"""
        # brightness
        image_array = image_array.astype('float32')
        image_array = ChartAugmenter.brightness(image_array, brightness_delta)
        # color jitter
        if np.random.randint(0, 2):
            image_array = ChartAugmenter.contrast(image_array, contrast_low, contrast_high)
            image_array = ChartAugmenter.saturation(image_array, saturation_low, saturation_high)
            image_array = ChartAugmenter.hue(image_array, hue_delta)
        else:
            image_array = ChartAugmenter.saturation(image_array, saturation_low, saturation_high)
            image_array = ChartAugmenter.hue(image_array, hue_delta)
            image_array = ChartAugmenter.contrast(image_array, contrast_low, contrast_high)
        image_array = np.clip(image_array, 0, 255)
        return image_array.astype('uint8')

    @staticmethod
    def brightness(image_array, delta, prop=0.5):
        """
        亮度畸变
        :param image_array:
        :param delta:
        :param prop:
        :return:
        """
        if np.random.uniform(0, 1) > prop:
            delta = np.random.uniform(-delta, delta)
            image_array += delta
        return image_array

    @staticmethod
    def contrast(image_array, low, high, prop=0.5):
        """
        对比度畸变
        :param image_array:
        :param low:
        :param high:
        :param prop:
        :return:
        """
        if np.random.uniform(0, 1) > prop:
            alpha = np.random.uniform(low, high)
            image_array *= alpha
        return image_array

    @staticmethod
    def saturation(image_array, low, high, prop=0.5):
        """
        饱和度畸变
        :param image_array:
        :param low:
        :param high:
        :param prop:
        :return:
        """
        if np.random.uniform(0, 1) > prop:
            alpha = np.random.uniform(low, high)
            gray = image_array * np.array([[[0.299, 0.587, 0.114]]])
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            image_array *= alpha
            image_array += gray
        return image_array

    @staticmethod
    def hue(image_array, delta, prop=0.5):
        """
        色相畸变
        :param image_array:
        :param delta:
        :param prop:
        :return:
        """
        if np.random.uniform(0, 1) > prop:
            alpha = np.random.uniform(-delta, delta)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            image_array = np.dot(image_array, np.array(t))
        return image_array
