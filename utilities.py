import skimage


def psnr(true_image,up_sampled_image):
    return skimage.metrics.peak_signal_noise_ratio(true_image, up_sampled_image)


def mse(true_image,up_sampled_image):
    return skimage.metrics.mean_squared_error(true_image, up_sampled_image)


def ssim(true_image,up_sampled_image):
    return skimage.metrics.structural_similarity(true_image,up_sampled_image)