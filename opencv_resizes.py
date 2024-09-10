import cv2
import matplotlib.pyplot as plt


def print_images(n_row, n_col, images):
    fig, axs = plt.subplots(n_row, n_col, figsize=(12, 8))
    axs = axs.flatten()

    for image, ax in zip(images, axs):
        ax.imshow(cv2.cvtColor(image['data'], cv2.COLOR_BGR2RGB))
        ax.set_title(image['name'])

    plt.tight_layout(pad=1)  # Adjust padding between and around images
    plt.show()


def edsr(image):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "opencv-models/EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 4)  # set the model by passing the value and the upsampling ratio
    result = sr.upsample(image)  # upscale the input image
    return result


def espcn(image):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "opencv-models/ESPCN_x4.pb"
    sr.readModel(path)
    sr.setModel("espcn", 4)  # set the model by passing the value and the upsampling ratio
    result = sr.upsample(image)  # upscale the input image
    return result


def lap_srn(image):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "opencv-models/LapSRN_x4.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 4)  # set the model by passing the value and the upsampling ratio
    result = sr.upsample(image)  # upscale the input image
    return result


def fsrcnn(image):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "opencv-models/FSRCNN_x4.pb"
    sr.readModel(path)
    sr.setModel("fsrcnn", 4)  # set the model by passing the value and the upsampling ratio
    result = sr.upsample(image)  # upscale the input image
    return result


def bicubic_resize(image, ratio=4):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)


def lanczos_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)


def laplace_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    first = cv2.pyrUp(image)
    return cv2.pyrUp(first)


if __name__ == "__main__":
    img = cv2.imread("butterfly.png")
    img_high_res = cv2.imread("butterfly_LR.png")

    espcn_res = {'data': espcn(img), 'name': 'ESPCN'}
    edsr_res = {'data': edsr(img), 'name': 'EDSR'}
    lap_srn_res = {'data': lap_srn(img), 'name': 'LapSRN'}
    fsrcnn_res = {'data': fsrcnn(img), 'name': 'FSRCNN'}
    lanczos_resize_res = {'data': lanczos_resize(img, 4), 'name': 'Lanczos'}
    laplace_resize_res = {'data': laplace_resize(img, 4), 'name': 'Laplace'}
    bicubic_resize_res = {'data': bicubic_resize(img, 4), 'name': 'Bicubic'}

    print_images(2, 3,
                 [{'data': img_high_res, 'name': 'HR'}, {'data': img, 'name': 'LR'}, bicubic_resize_res,
                  lanczos_resize_res,
                  fsrcnn_res, edsr_res])
