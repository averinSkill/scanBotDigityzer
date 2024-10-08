
from pathlib import Path
import cv2
import numpy as np
import ezdxf
from ezdxf import units
# from ezdxf.addons.drawing import Frontend, RenderContext, pymupdf, layout, config

from pathlib import Path
from skimage.morphology import skeletonize
# from skimage.measure import approximate_polygon, subdivide_polygon

# from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

import pillow_heif



def order_points(pts):
    # Инициализация списка координат, которые будут упорядочены
    # таким образом: верхний левый, верхний правый, нижний правый, нижний левый
    rect = np.zeros((4, 2), dtype="float32")

    # Верхний левый угол будет иметь наименьшую сумму,
    # нижний правый угол будет иметь наибольшую сумму
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Верхний правый угол будет иметь наименьшую разность,
    # нижний левый будет иметь наибольшую разность
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Возвращаем упорядоченные координаты
    return rect


def four_point_transform(image, pts):
    # Получаем упорядоченные координаты и применяем перспективное преобразование
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Вычисляем ширину нового изображения
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    a = min(int(widthA), int(widthB))
    maxWidth = max(int(widthA), int(widthB))

    # Вычисляем высоту нового изображения
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxHeight > maxWidth:
        # Высота больше ширины, фиксируем высоту и вычисляем ширину
        H = maxHeight
        aspect_ratio = 1.25
        W = int(H / aspect_ratio)
    else:
        # Ширина больше высоты, фиксируем ширину и вычисляем высоту

        H = maxWidth
        aspect_ratio = 0.8
        W = int(H / aspect_ratio)

    # Теперь у нас есть координаты нового изображения, применяем перспективное преобразование
    dst = np.array([
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1]], dtype="float32")

    # Матрица перспективного преобразования
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (W, H))
    #     print(warped.shape)
    #     # Закрашивает по границе по 10 пикселей цветом, который в углах
    m, n = warped.shape[:2]
    left_up_hsv = np.mean(np.mean(warped[:10, :10], axis=0), axis=0).astype(int)
    left_down_hsv = np.mean(np.mean(warped[m - 10:, :10], axis=0), axis=0).astype(int)
    right_up_hsv = np.mean(np.mean(warped[:10, n - 10:], axis=0), axis=0).astype(int)
    right_down_hsv = np.mean(np.mean(warped[m - 10:, n - 10:], axis=0), axis=0).astype(int)
    warped[:10, :n] = left_up_hsv
    warped[:m, :10] = left_down_hsv
    warped[m - 10:, :n] = right_up_hsv
    warped[:m, n - 10:] = right_down_hsv

    # Возвращаем преобразованное изображение
    return warped


# Возвращает восстановленное и обрезанное по контуру подложки изображение (фон+лекало)
def bgr_finder_super_new(filename, threshold=150):
    print("filename.suffix= ", filename.suffix.lower())
    if filename.suffix.lower() == ".heic":
        heif_file = pillow_heif.open_heif(filename, convert_hdr_to_8bit=False, bgr_mode=True)
        image = np.asarray(heif_file)
    else:
        image = cv2.imread(filename)
    #     print("bgr_finder_super_new image.shape=", image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_pic, w_pic = image.shape[:2]

    _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    #     cv2.imwrite(filename.replace("IMG","bgr_finder_new_thresh_1"), thresh1)
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Перебор контуров и поиск четырехугольника, который предположительно является листом
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    warped = four_point_transform(image, approx.reshape(4, 2))
    warped = cv2.detailEnhance(warped, sigma_s=10, sigma_r=0.15)
    _, thresh2 = cv2.threshold(warped, threshold, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(filename.replace("IMG", "bgr_finder_super_new_thresh2"), thresh2)
    # Creating the kernel(2d convolution matrix)
    #     thresh2  = cv2.GaussianBlur(thresh2,(7,7),0)
    thresh_gray = cv2.cvtColor(thresh2, cv2.COLOR_BGR2GRAY)

    #     cv2.imwrite(filename.replace("IMG","bgr_finder_super_new_thresh_gray"), thresh_gray)
    perimeter = warped.shape[0] * 2 + warped.shape[1] * 2
    pxl_v_mm = perimeter / (1005 * 2 + 804 * 2)
    #     pxl_v_mm = perimeter / (1000 * 2 + 800 * 2)
    return thresh_gray, pxl_v_mm, h_pic


# Возвращает внешний контур лекала
def get_lecalo_contour(warped):
    #     print("get_lecalo_contour np.info(warped) = ", warped.shape)
    #     mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_blured = cv2.blur(warped, (10, 10))
    _, thresh = cv2.threshold(warped_blured, 180, 255, cv2.THRESH_BINARY_INV)
    cont_VNESHNII, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)  # !!!!!!!!!!!!!!!!ВНЕШНИЙ КОНТУР!!!!!!!!!!!!!!
    cont_VNESHNII = sorted(cont_VNESHNII, key=cv2.contourArea, reverse=True)[0]

    #     new = cv2.drawContours(warped, cont_VNESHNII, -1, (0, 0, 0), 2)

    #     cv2.imwrite("lecalo_contour.png", new)
    return np.array(cont_VNESHNII).squeeze()


#

def fill_contour(image, coords, type=0):
    '''
     type_cont= 0 - inside coord
     type_cont= 1 - outside coord
    '''

    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([coords], dtype=np.int32)
    try:
        channel_count = image.shape[2]
    except:
        channel_count = 1

    ignore_mask_color = (255,) * channel_count

    filpol = cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    if type == 0:
        masked_image = cv2.bitwise_and(image, filpol)
    elif type == 1:
        mask_inv = cv2.bitwise_not(filpol)
        masked_image = cv2.bitwise_and(image, mask_inv)

    return masked_image


def get_skeleton_iamge(threshold_image):
    threshold_image = cv2.blur(threshold_image, (10, 10))
    skeleton = skeletonize(threshold_image / 255)
    skeleton = skeleton.astype(np.uint8)
    skeleton *= 255
    return skeleton


def get_increase_contour(contour, pixels):
    # Вычисляем центр контура
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return contour
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Увеличиваем каждую точку контура
    increased_contour = np.zeros_like(contour)
    for i in range(len(contour)):
        x, y = contour[i]
        angle = np.arctan2(y - cy, x - cx)
        x += np.cos(angle) * pixels
        y += np.sin(angle) * pixels
        increased_contour[i] = np.array([x, y], dtype=np.int32)

    return increased_contour


def get_smoothed_contour(cont_VNESHNII):
    #     print("get_smoothed_contour - ", len(cont_VNESHNII))

    # Преобразуем контур в массив координат
    contour_coords = np.array(cont_VNESHNII).squeeze()
    if len(cont_VNESHNII) > 36:
        hull = ConvexHull(contour_coords)
        vertices = contour_coords[hull.vertices]
        w = np.ones(contour_coords.shape[0])
        w[hull.vertices] = 2

        # Разделяем координаты на отдельные массивы
        x = contour_coords[:, 0]
        y = contour_coords[:, 1]

        # Параметрическая интерполяция контура
        tck, u = splprep([x, y], s=1, w=w)
        u_new = np.linspace(u.min(), u.max(), int(contour_coords.shape[0] / 4))
        x_new, y_new = splev(u_new, tck)
        # print("length linspace=", int(contour_coords.shape[0]/5))

        return np.vstack((x_new, y_new)).T
    else:
        return contour_coords


def digityzer(filename, threshold):
    fln = Path(str(filename).replace("Photo", "Result"))
    fln = fln.with_name(fln.stem + "_result")
    # print("DIGITYZER: filename= ", filename)
    # print("DIGITYZER: threshold= ", threshold)
    warped_, pxl_v_mm, h = bgr_finder_super_new(filename, threshold)
    # print("DIGITYZER: pxl_v_mm= ", pxl_v_mm)

    cont_VNESHNII = get_lecalo_contour(warped_)
    cont_outside_final = get_smoothed_contour(cont_VNESHNII)

    cont_minus = get_increase_contour(cont_VNESHNII, -10)
    crop_img = fill_contour(warped_, cont_minus, type=0)

    dwg = ezdxf.new("R2010")
    dwg.units = units.MM
    dwg.header['$INSUNITS'] = units.MM
    msp = dwg.modelspace()

    # Добавляем внешний контур в dxf
    points_mm_vneshnii = [(point[0] / pxl_v_mm, (h - point[1]) / pxl_v_mm) for point in cont_outside_final]
    msp.add_lwpolyline(points_mm_vneshnii, dxfattribs={"layer": "0", "lineweight": 1})

    # выделяет все контуры на crop_img и записывает их в dxf
    contours, _ = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cnt = np.squeeze(cnt, axis=1)
            points_mm = [(pointi[0] / pxl_v_mm, (h - pointi[1]) / pxl_v_mm) for pointi in cnt]
            msp.add_lwpolyline(points_mm, dxfattribs={"layer": "inside", "lineweight": 1})

    # filename = f"threshold_{threshold}_" + filename
    # dwg.saveas(filename.replace("jpg", "dxf"))
    dwg.saveas(fln.with_suffix('.dxf'))
    print("Файл сохранен - ", fln.with_suffix('.dxf'))

    return fln.with_suffix('.dxf')

