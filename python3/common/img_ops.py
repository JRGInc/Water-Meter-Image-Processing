__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import colorsys
import cv2
import imutils
import logging
import math
import numpy as np
import os.path
import random
from PIL import Image, ImageFont, ImageDraw

logfile = 'januswm-capture'
logger = logging.getLogger(logfile)


class ShapeDetector:
    def __init__(self):
        pass

    @staticmethod
    def detect(c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape


def reduce(
        img_orig_stream,
        img_orig_url: str,
        img_dest_url: str,
        img_dest_qual: int = 100,
) -> (bool, str):
    """
    Crops and saves given image

    :param img_orig_stream
    :param img_orig_url: str
    :param img_dest_url: str
    :param img_dest_qual: int

    :return img_redx_err: bool
    """
    img_redx_err = False
    img_orig = None

    try:
        if img_orig_stream is not None:
            img_orig = Image.open(fp=img_orig_stream)
        elif os.path.isfile(path=img_orig_url):
            img_orig = Image.open(fp=img_orig_url)
        else:
            img_redx_err = True
            log = 'OS failed to locate image {0} to save.'. \
                format(img_orig_url)
            logger.error(msg=log)
            print(log)

        if not img_redx_err:
            img_orig.save(
                fp=img_dest_url,
                format='jpeg',
                optimize=True,
                quality=img_dest_qual
            )
            img_orig.close()

            log = 'PIL successfully reduced image {0}.'. \
                format(img_dest_url)
            logger.info(msg=log)

    except Exception as exc:
        img_redx_err = True
        log = 'PIL failed to reduce image {0}.'.format(img_dest_url)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return img_redx_err


def scale(
        img_orig_url: str,
        img_scale_url: str,
) -> tuple:
    """
    Finds locations and sizes of screw heads and center needle pivot for given image

    :param img_orig_url: str
    :param img_scale_url: str

    :return img_scale: list
    :return img_scale_err: bool
    """
    img_scale_err = False
    img_scale = None

    if os.path.isfile(path=img_orig_url):

        try:
            img_orig = cv2.imread(filename=img_orig_url)
            img_scale = img_orig
            img_orig_h = img_orig.shape[0]
            img_orig_w = img_orig.shape[1]

            # Scale image to 1920x1080
            if img_orig_h == 480:
                scale_factor = 2.250
                width = int(img_orig_w * scale_factor)
                height = int(img_orig_h * scale_factor)
                img_scale = cv2.resize(
                    src=img_orig,
                    dsize=(width, height)
                )
            elif img_orig_h == 720:
                scale_factor = 1.500
                width = int(img_orig_w * scale_factor)
                height = int(img_orig_h * scale_factor)
                img_scale = cv2.resize(
                    src=img_orig,
                    dsize=(width, height)
                )
            elif img_orig_h == 922:
                scale_factor = 1.171
                width = int(img_orig_w * scale_factor)
                height = int(img_orig_h * scale_factor)
                img_scale = cv2.resize(
                    src=img_orig,
                    dsize=(width, height)
                )
            elif img_orig_h == 1080:
                pass
            elif img_orig_h == 1536:
                pass
            elif img_orig_h == 2464:
                pass

            cv2.imwrite(
                filename=img_scale_url,
                img=img_scale,
                params=[
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    100
                ]
            )

        except Exception as exc:
            img_scale_err = True
            log = 'OpenCV failed to scale image {0}.'. \
                format(img_scale_url)
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_scale_err = True
        log = 'OS failed to locate image {0} to scale.'. \
            format(img_orig_url)
        logger.error(msg=log)
        print(log)

    return img_scale, img_scale_err


def find_glass(
        img_orig,
        img_glass_url: str,
) -> tuple:
    """
    Finds the glass perimeter, glass center, and crops

    :param img_orig: opencv image
    :param img_glass_url: str

    :return img_glass: list
    :return img_glass_err: bool
    """
    img_glass = None
    img_glass_err = False
    min_radius = 450
    max_radius = 550

    if img_orig is not None:
        try:
            img_gray = cv2.cvtColor(
                src=img_orig,
                code=cv2.COLOR_BGR2GRAY
            )
            # thresh, img_thresh = cv2.threshold(
            #     src=img_gray,
            #     thresh=45,
            #     maxval=255,
            #     type=cv2.THRESH_BINARY
            # )
            # cv2.imwrite(
            #     filename=img_glass_url + '-thresh.jpg',
            #     img=img_thresh,
            #     params=[
            #         int(cv2.IMWRITE_JPEG_QUALITY),
            #         100
            #     ]
            # )
            img_blur = cv2.medianBlur(
                src=img_gray,
                ksize=5
            )

            circles = cv2.HoughCircles(
                image=img_blur,
                method=cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=2000,
                param1=100,
                param2=20,
                minRadius=min_radius,
                maxRadius=max_radius
            )

            if circles is not None:
                img_glass = circles[0][0]
                print('Number of circles: {0}'.format(len(circles[0])))
                # for circle in range(0, len(circles[0])):
                #
                #     # copy grayed image to edges folder than perform this step to save image with overlaid lines,
                #     # otherwise unnecessary
                #     cv2.circle(
                #         img=img_orig,
                #         center=(
                #             circles[0][circle][0],
                #             circles[0][circle][1]
                #         ),
                #         radius=circles[0][circle][2],
                #         color=(0, 0, 255),
                #         thickness=2
                #     )

                # cv2.imwrite(
                #     filename=img_glass_url,
                #     img=img_orig,
                #     params=[
                #         int(cv2.IMWRITE_JPEG_QUALITY),
                #         100
                #     ]
                # )

                log = 'OpenCV successfully found glass circles.'
                logger.error(msg=log)
                print(log)

            else:
                img_glass_err = True
                log = 'OpenCV failed to glass circles.'
                logger.error(msg=log)
                print(log)

        except Exception as exc:
            img_glass_err = True
            log = 'OpenCV failed to glass circles.'
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_glass_err = True
        log = 'Original image to locate glass does not exist.'
        logger.error(msg=log)
        print(log)

    return img_glass, img_glass_err


def crop_glass(
        img_orig,
        img_glass: list,
        img_square_url: str,
) -> tuple:
    """
    Crops digit rectangle for given image

    :param img_orig: opencv image
    :param img_glass: list
    :param img_square_url: str

    :return img_square: opencv image
    :return img_square_err: bool
    """
    img_square = None
    img_square_err = False

    if img_orig is not None:
        try:

            img_crop_dict = {
                'ulx': int(img_glass[0] - img_glass[2]),
                'uly': int(img_glass[1] - img_glass[2]),
                'brx': int(img_glass[0] + img_glass[2]),
                'bry': int(img_glass[1] + img_glass[2])
            }
            img_square = img_orig[
                         img_crop_dict['uly']:img_crop_dict['bry'],
                         img_crop_dict['ulx']:img_crop_dict['brx']
                         ]

            cv2.imwrite(
                filename=img_square_url,
                img=img_square,
                params=[
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    100
                ]
            )

            log = 'Calculated square window at:      ' + \
                  '{0}, {1}, {2}, {3} (upper, lower, left, right).'. \
                      format(img_crop_dict['uly'], img_crop_dict['bry'], img_crop_dict['ulx'], img_crop_dict['brx'])
            logger.info(msg=log)
            print(log)

            log = 'Successfully cropped glass in image: {0}'. \
                format(img_square_url)
            logger.info(msg=log)
            print(log)

        except Exception as exc:
            img_square_err = True
            log = 'Failed to crop glass in image {0}.'. \
                format(img_square_url)
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_square_err = True
        log = 'Original image to crop glass does not exist.'
        logger.error(msg=log)
        print(log)

    return img_square, img_square_err


def find_angles(
        img_orig_url: str,
        bbox_dict: dict
) -> tuple:
    """
    Finds locations and sizes of screw heads and center needle pivot for given image

    :param img_orig_url: str
    :param bbox_dict: dict

    :return img_screws: list
    :return img_screw_err: bool
    """
    img_ang_err = False
    img_ang_list = []

    if os.path.isfile(path=img_orig_url):
        try:
            img_ctr_x = int((bbox_dict['pivot'][1] + bbox_dict['pivot'][3]) / 2)
            img_ctr_y = int((bbox_dict['pivot'][2] + bbox_dict['pivot'][4]) / 2)
            screw_ctr_x = int((bbox_dict['screw'][1] + bbox_dict['screw'][3]) / 2)
            screw_ctr_y = int((bbox_dict['screw'][2] + bbox_dict['screw'][4]) / 2)
            digw_ctr_x = int((bbox_dict['digits'][1] + bbox_dict['digits'][3]) / 2)
            digw_ctr_y = int((bbox_dict['digits'][2] + bbox_dict['digits'][4]) / 2)

            # Calculate angle for bottom edge of digit window
            digw_ang = math.degrees(
                math.atan2(
                    digw_ctr_y - img_ctr_y,
                    digw_ctr_x - img_ctr_x
                )
            )
            digw_ang = round(digw_ang, 2)
            if digw_ang < 0:
                digw_ang = 360 + digw_ang

            # Calculate angle for bottom edge of digit window
            screw_ang = math.degrees(
                math.atan2(
                    screw_ctr_y - img_ctr_y,
                    screw_ctr_x - img_ctr_x
                )
            )
            screw_ang = round(screw_ang, 2)
            if screw_ang < 0:
                screw_ang = 360 + screw_ang

            diff_ang = digw_ang - screw_ang

            grot_ang = 0
            if (-105 <= diff_ang <= -75) or (255 <= diff_ang <= 285):
                grot_ang = 180 + screw_ang
            elif (75 <= diff_ang <= 105) or (-285 <= diff_ang <= -255):
                grot_ang = screw_ang

            img_ang_list = [img_ctr_x, img_ctr_y, grot_ang, diff_ang, screw_ang, digw_ang]

        except Exception as exc:
            img_ang_err = True
            log = 'OpenCV failed to determine angles for image {0}.'.format(img_orig_url)
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_ang_err = True
        log = 'OS failed to locate image {0} to rotate.'. \
            format(img_orig_url)
        logger.error(msg=log)
        print(log)

    return img_ang_err, img_ang_list


def rotate(
        img_orig_url: str,
        img_grotd_url: str,
        img_frotd_url: str,
        img_ang_list: list
) -> tuple:
    """
    Rotates and saves given image based on bottom edge of digit window, fine rotation

    :param img_orig_url: str
    :param img_grotd_url: str
    :param img_frotd_url: str
    :param img_ang_list: list

    :return img_rotd: opencv image
    :return img_rotd_err: bool
    """
    img_frotd = None
    img_rotd_err = False

    if os.path.isfile(path=img_orig_url):
        try:

            img_orig = cv2.imread(filename=img_orig_url)
            img_orig_h = img_orig.shape[0]
            img_orig_w = img_orig.shape[1]

            m_grotd = cv2.getRotationMatrix2D(
                center=(
                    img_ang_list[0],
                    img_ang_list[1]
                ),
                angle=img_ang_list[2],
                scale=1.0
            )
            img_grotd = cv2.warpAffine(
                src=img_orig,
                M=m_grotd,
                dsize=(img_orig_w, img_orig_h)
            )
            cv2.imwrite(
                filename=img_grotd_url,
                img=img_grotd,
                params=[
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    100
                ]
            )

            log = 'Successful gross rotation of image {0} by angle {1}.'. \
                format(img_grotd_url, img_ang_list[2])
            logger.info(msg=log)
            print(log)

            img_crop_dict = None
            if img_orig_h == 1536:
                img_crop_dict = {
                    'ulx': img_ang_list[0] - 240,
                    'uly': img_ang_list[1] + 95,
                    'brx': img_ang_list[0] + 240,
                    'bry': img_ang_list[1] + 205
                }
            elif img_orig_h == 2464:
                img_crop_dict = {
                    'ulx': img_ang_list[0] - 390,
                    'uly': img_ang_list[1] + 140,
                    'brx': img_ang_list[0] + 390,
                    'bry': img_ang_list[1] + 310
                }

            img_rect = img_grotd[
                img_crop_dict['uly']:img_crop_dict['bry'],
                img_crop_dict['ulx']:img_crop_dict['brx']
            ]
            img_h = img_rect.shape[0]

            img_gray = cv2.cvtColor(
                src=img_rect,
                code=cv2.COLOR_BGR2GRAY
            )
            thresh, img_thresh = cv2.threshold(
                src=img_gray,
                thresh=80,
                maxval=255,
                type=cv2.THRESH_BINARY_INV
            )
            # cv2.imwrite(
            #     filename=img_frotd_url + '-thresh.jpg',
            #     img=img_thresh,
            #     params=[
            #         int(cv2.IMWRITE_JPEG_QUALITY),
            #         100
            #     ]
            # )

            l_lower = 0
            r_lower = 0

            left = 0
            right = 0

            if img_orig_h == 1536:
                left = 90
                right = 380
            elif img_orig_h == 2464:
                left = 150
                right = 620

            for y_pix in range((img_h - 1), 0, -1):
                if img_thresh[y_pix][left] == 255:
                    l_lower = y_pix
                    break

            for y_pix in range((img_h - 1), 0, -1):
                if img_thresh[y_pix][right] == 255:
                    r_lower = y_pix
                    break

            # print('LEFT COORDS: {0}, {1}'.format(left, l_lower))
            # print('RIGHT COORDS: {0}, {1}'.format(right, r_lower))

            img_rect_ang = math.degrees(
                math.atan2(
                    r_lower - l_lower,
                    right - left
                )
            )
            img_rect_ang = round(img_rect_ang, 2)

            img_frotd_ctr = [left, l_lower]

            img_rect_h = img_rect.shape[0]
            img_rect_w = img_rect.shape[1]
            m_frotd = cv2.getRotationMatrix2D(
                center=(
                    img_frotd_ctr[0],
                    img_frotd_ctr[1]
                ),
                angle=img_rect_ang,
                scale=1.0
            )
            img_frotd = cv2.warpAffine(
                src=img_rect,
                M=m_frotd,
                dsize=(img_rect_w, img_rect_h)
            )
            cv2.imwrite(
                filename=img_frotd_url,
                img=img_frotd,
                params=[
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    100
                ]
            )

            log = 'Successful fine rotation of image {0} by angle {1}.'. \
                format(img_frotd_url, img_rect_ang)
            logger.info(msg=log)
            print(log)

        except Exception as exc:
            img_rotd_err = True
            log = 'Failed to rotate image {0}.'.format(img_orig_url)
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_rotd_err = True
        log = 'OS failed to locate image {0} to rotate.'. \
            format(img_orig_url)
        logger.error(msg=log)
        print(log)

    return img_rotd_err, img_frotd


def crop_rect(
        img_rotd,
        img_rect_url: str,
        img_digw_url: str,
        tf_dict: dict,
) -> tuple:
    """
    Crops digit rectangle for given image

    :param img_rotd: opencv image
    :param img_rect_url: str
    :param img_digw_url: str
    :param tf_dict: dict

    :return img_digw: opencv image
    :return img_rect_err: bool
    """
    img_digw = None
    img_rect_err = False
    upper = 0
    lower = 0
    left = 0
    right = 0

    tf_lower = tf_dict['shadow'] + tf_dict['height']

    if img_rotd is not None:
        try:
            img_rotd_h = img_rotd.shape[0]
            img_rotd_w = img_rotd.shape[1]
            img_gray = cv2.cvtColor(
                src=img_rotd,
                code=cv2.COLOR_BGR2GRAY
            )
            thresh, img_thresh = cv2.threshold(
                src=img_gray,
                thresh=80,
                maxval=255,
                type=cv2.THRESH_BINARY_INV
            )

            for y_pix in range(0, img_rotd_h):
                if img_thresh[y_pix][int(img_rotd_w / 2)] == 255:
                    upper = y_pix
                    break

            for y_pix in range((img_rotd_h - 1), 0, -1):
                if (img_thresh[y_pix][int(img_rotd_w / 2)] == 255) and \
                        (img_thresh[y_pix - 5][int(img_rotd_w / 2)] == 255):
                    lower = y_pix
                    break

            for x_pix in range(0, img_rotd_w):
                # Five checks to avoid false positive with needle in rest position
                if (img_thresh[lower - 5][x_pix] == 255) and \
                        (img_thresh[lower - 5][x_pix + 10] == 255) and \
                        (img_thresh[lower - 5][x_pix + 20] == 255) and \
                        (img_thresh[lower - 5][x_pix + 30] == 255):
                    left = x_pix
                    break

            for x_pix in range((img_rotd_w - 1), 0, -1):
                if img_thresh[lower - 40][x_pix] == 255:
                    right = x_pix
                    break

            log = 'Calculated raw digit window edges at:      ' + \
                  '{0}, {1}, {2}, {3} (upper, lower, left, right).'.format(upper, lower, left, right)
            logger.info(msg=log)
            print(log)

            ulx0 = left + 1
            uly0 = upper + 2
            brx0 = right - 23  # Crop off non-moving bushing from right edge
            bry0 = lower - 1
            img_h = bry0 - uly0

            # Make image width evenly divisible by 6
            length_rem = (brx0 - ulx0) % 6
            if (length_rem >= 1) and (length_rem < 4):
                brx0 -= length_rem
            elif (length_rem >= 4) and (length_rem < 6):
                brx0 += 6 - length_rem

            log = 'Calculated adjusted digit window edges at: ' + \
                  '{0}, {1}, {2}, {3} (upper, lower, left, right).'.format(uly0, bry0, ulx0, brx0)
            logger.info(msg=log)
            print(log)

            img_crop_dict1 = {
                'ulx': ulx0,
                'uly': uly0,
                'brx': brx0,
                'bry': bry0
            }
            img_digw = img_rotd[
                       img_crop_dict1['uly']:img_crop_dict1['bry'],
                       img_crop_dict1['ulx']:img_crop_dict1['brx']
                       ]

            # If Image height is less than tensor flow requirements, pad the upper edge
            # with sufficient rows to create proper sized digit image and add one row
            # to bottom edge.
            # if img_h < tf_lower:
            #     top = tf_lower - img_h
            #     img_digw = cv2.copyMakeBorder(
            #         src=img_digw,
            #         top=top,
            #         bottom=0,
            #         left=0,
            #         right=0,
            #         borderType=cv2.BORDER_CONSTANT,
            #         value=(0, 0, 0)
            #     )
            #     img_digw = cv2.copyMakeBorder(
            #         src=img_digw,
            #         top=0,
            #         bottom=1,
            #         left=0,
            #         right=0,
            #         borderType=cv2.BORDER_REPLICATE
            #     )
            #
            #     log = 'Added {0} rows to top and 1 row to bottom of {1}.'. \
            #         format(top, img_digw_url)
            #     logger.info(msg=log)
            #     print(log)

            cv2.imwrite(
                filename=img_digw_url,
                img=img_digw,
                params=[
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    100
                ]
            )

            cv2.line(
                img=img_rotd,
                pt1=(0, (uly0 - 1)),
                pt2=(img_rotd_w, (uly0 - 1)),
                color=(0, 255, 0),
                thickness=1
            )
            cv2.line(
                img=img_rotd,
                pt1=(0, bry0),
                pt2=(img_rotd_w, bry0),
                color=(0, 255, 0),
                thickness=1
            )

            cv2.line(
                img=img_rotd,
                pt1=((ulx0 - 1), 0),
                pt2=((ulx0 - 1), img_rotd_h),
                color=(0, 255, 0),
                thickness=1
            )
            cv2.line(
                img=img_rotd,
                pt1=(brx0, 0),
                pt2=(brx0, img_rotd_h),
                color=(0, 255, 0),
                thickness=1
            )

            cv2.imwrite(
                filename=img_rect_url,
                img=img_rotd,
                params=[
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    100
                ]
            )

            log = 'Successfully cropped digit window in image: {0}'. \
                format(img_digw_url)
            logger.info(msg=log)
            print(log)

        except Exception as exc:
            img_rect_err = True
            log = 'Failed to determine digit window in image {0}.'. \
                format(img_digw_url)
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_rect_err = True
        log = 'Rotated image to crop digit window does not exist.'
        logger.error(msg=log)
        print(log)

    return img_digw, img_rect_err


def crop_digits(
        img_digw,
        img_digw_url: str,
        img_path_dict: dict,
        tf_dict: dict,
        mode_str: str
) -> bool:
    """
    Crops and saves given image as separate digits

    :param img_digw: opencv image
    :param img_digw_url: str
    :param img_path_dict: dict
    :param tf_dict: dict
    :param mode_str: str

    :return img_digs_err: bool
    """
    img_digs_err = False

    # img_upper = tf_dict['shadow']
    img_upper = 0
    # img_lower = tf_dict['shadow'] + tf_dict['height']
    if mode_str == 'train':
        if (tf_dict['full_width'] % 2) == 0:
            img_left = int(tf_dict['full_width'] / 2)
            img_right = int(tf_dict['full_width'] / 2)
        else:
            img_left = int((tf_dict['full_width'] / 2) + 0.5)
            img_right = int((tf_dict['full_width'] / 2) - 0.5)

    else:
        if (tf_dict['tf_width'] % 2) == 0:
            img_left = int(tf_dict['tf_width'] / 2)
            img_right = int(tf_dict['tf_width'] / 2)
        else:
            img_left = int((tf_dict['tf_width'] / 2) + 0.5)
            img_right = int((tf_dict['tf_width'] / 2) - 0.5)

    if img_digw is not None:
        img_inv_url = os.path.join(
            img_path_dict['inv'],
            'inv' + os.path.basename(img_digw_url)[4::]
        )

        try:
            img_h = img_digw.shape[0]
            img_lower = img_h - 1
            img_w = img_digw.shape[1]
            dig_w = int(img_w / 6)
            dig_w_ctr = int(dig_w / 2)
            log = 'Raw digit width is {0} pixels and center is {1} pixels.'. \
                format(dig_w, dig_w_ctr)
            logger.info(msg=log)
            print(log)

            # Try running threshold and edge detection on just brighter objects
            # then subtracting from other image
            img_gray = cv2.cvtColor(
                src=img_digw,
                code=cv2.COLOR_BGR2GRAY
            )
            # 125 original
            thresh, img_thresh = cv2.threshold(
                src=img_gray,
                thresh=75,
                maxval=255,
                type=cv2.THRESH_BINARY_INV
            )
            cv2.imwrite(
                filename=img_inv_url,
                img=img_thresh,
                params=[
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    100
                ]
            )
            img_edge = cv2.Canny(
                image=img_thresh,
                threshold1=30,
                threshold2=200
            )

            for digit in range(0, 6):
                img_dig_url = os.path.join(
                    img_path_dict['digs'],
                    'digs' + '_d' + str(5 - digit) + os.path.basename(img_digw_url)[4::]
                )
                img_tdig_url = os.path.join(
                    img_path_dict['pred'],
                    'pred' + '_d' + str(5 - digit) + os.path.basename(img_digw_url)[4::]
                )
                img_cont_url = os.path.join(
                    img_path_dict['cont'],
                    'cont' + '_d' + str(5 - digit) + os.path.basename(img_digw_url)[4::]
                )

                start_x = digit * dig_w
                end_x = start_x + dig_w

                img_dig_orig = img_digw[0:img_h, start_x:end_x]
                img_dig_gray = img_gray[0:img_h, start_x:end_x]
                contours, hierarchy = cv2.findContours(
                    image=img_edge[0:img_h, start_x:end_x],
                    mode=cv2.RETR_EXTERNAL,
                    method=cv2.CHAIN_APPROX_NONE
                )
                img_dig_cnt = cv2.drawContours(
                    image=img_dig_orig,
                    contours=contours,
                    contourIdx=-1,
                    color=(255, 0, 255),
                    thickness=1
                )

                # Must find contour of greatest width and set tensor window accordingly
                # Perhaps set cnt_x to right side of frame and test for cnt_x <= cnt_x
                cnt_w = 0
                cnt_x = 0
                for contour in range(0, len(contours)):
                    x, y, w, h = cv2.boundingRect(contours[contour])
                    if (w > 10) and (w > cnt_w):
                        cnt_w = w
                        cnt_x = x

                cnt_ctr = cnt_x + int(cnt_w / 2)

                log = 'Digit {0} tensor flow raw horizontal boundaries '.format(digit) + \
                      'are:     {0}, {1}, and {2} (left, center, right).'.format(cnt_x, cnt_ctr, (cnt_x + cnt_w))
                logger.info(msg=log)
                print(log)

                cv2.line(
                    img=img_dig_cnt,
                    pt1=(0, img_upper),
                    pt2=(img_w, img_upper),
                    color=(0, 255, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(0, img_lower),
                    pt2=(img_w, img_lower),
                    color=(0, 255, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(cnt_x, 0),
                    pt2=(cnt_x, img_h),
                    color=(0, 255, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(cnt_ctr, 0),
                    pt2=(cnt_ctr, img_h),
                    color=(0, 0, 255),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=((cnt_x + cnt_w), 0),
                    pt2=((cnt_x + cnt_w), img_h),
                    color=(0, 255, 0),
                    thickness=1
                )

                # If contour width is over 48, very likely due to glare on digit drum
                # Adjust center line of digit to account for this glare
                if cnt_w >= 48:
                    # Glare skewed to right
                    # Move central point to left
                    if cnt_ctr > dig_w_ctr:
                        cnt_ctr = cnt_x + 24

                    # Glare skewed to left
                    # Move left edge to right full difference
                    elif cnt_ctr < dig_w_ctr:
                        cnt_ctr = cnt_x + cnt_w - 24

                # left = cnt_ctr - img_left
                # right = cnt_ctr + img_right
                left = cnt_x - 3
                right = cnt_x + cnt_w + 3

                # if contour is erroneously left-shifted
                if left < 0:
                    left = 0
                    # right = dig_w_ctr + img_right

                # if contour is erroneously right-shifted
                elif right >= dig_w:
                    # left = dig_w_ctr - img_left
                    right = dig_w - 1

                log = 'Digit {0} tensor flow adjusted horizontal boundaries '.format(digit) + \
                      'are: {0}, {1}, and {2} (left, center, right).'.format(left, cnt_ctr, right)
                logger.info(msg=log)
                print(log)

                cv2.line(
                    img=img_dig_cnt,
                    pt1=(left, 0),
                    pt2=(left, img_h),
                    color=(255, 0, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(right, 0),
                    pt2=(right, img_h),
                    color=(255, 0, 0),
                    thickness=1
                )
                cv2.imwrite(
                    filename=img_cont_url,
                    img=img_dig_cnt,
                    params=[
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        100
                    ]
                )

                if mode_str == 'train':
                    print('CROP IMAGE SAVE DIGITS')
                    cv2.imwrite(
                        filename=img_dig_url,
                        img=img_dig_gray[img_upper:img_lower, left:right],
                        params=[
                            int(cv2.IMWRITE_JPEG_QUALITY),
                            100
                        ]
                    )

                elif mode_str == 'test':
                    cv2.imwrite(
                        filename=img_dig_url,
                        img=img_dig_gray[img_upper:img_lower, left:right],
                        params=[
                            int(cv2.IMWRITE_JPEG_QUALITY),
                            100
                        ]
                    )

                elif mode_str == 'pred':
                    cv2.imwrite(
                        filename=img_tdig_url,
                        img=img_dig_gray[img_upper:img_lower, left:right],
                        params=[
                            int(cv2.IMWRITE_JPEG_QUALITY),
                            100
                        ]
                    )

            log = 'Successfully cropped digits from {0}.'. \
                format(img_inv_url)
            logger.info(msg=log)
            print(log)

        except Exception as exc:
            img_digs_err = True
            log = 'Failed to crop digits from {0}.'. \
                format(img_inv_url)
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_digs_err = True
        log = 'Digit window image to crop digits does not exist.'
        logger.error(msg=log)
        print(log)

    return img_digs_err


def overlay(
        img_digw_url: str,
        img_olay_url: str,
        img_olay_text: str
) -> bool:
    """
    Adds meter value banner to bottom of given image and saves

    :param img_digw_url: str
    :param img_olay_url: str
    :param img_olay_text: str

    :return img_olay_err: bool
    """
    img_olay_err = False

    if os.path.isfile(path=img_digw_url):
        try:
            # Open image, add blank banner, save, and close image
            img_digw = Image.open(fp=img_digw_url)
            brx, bry = img_digw.size
            img_olay = Image.new(
                mode='RGB',
                size=(brx, (bry + 30)),
                color=(0, 0, 0)
            )
            img_olay.paste(
                im=img_digw,
                box=(0, 0)
            )
            img_olay.save(
                fp=img_olay_url,
                format='jpeg',
                optimize=True,
                quality=100
            )
            img_digw.close()
            img_olay.close()

            # Open image, add text, save, and close image
            img_olay = Image.open(fp=img_olay_url)
            img_olay_draw = ImageDraw.Draw(im=img_olay)
            img_olay_font = ImageFont.truetype(
                font="DejaVuSans.ttf",
                size=18
            )
            img_olay_draw.text(
                xy=(10, (bry + 4)),
                text=img_olay_text,
                fill=(255, 255, 0, 255),
                font=img_olay_font
            )
            img_olay.save(
                fp=img_olay_url,
                format='jpeg',
                optimize=True,
                quality=100
            )
            img_olay.close()

            log = 'PIL successfully overlaid image {0}.'. \
                format(img_olay_url)
            logger.info(msg=log)

        except Exception as exc:
            img_olay_err = True
            log = 'PIL failed to overlay image {0}.'. \
                format(img_olay_url)
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_olay_err = True
        log = 'OS failed to locate image {0} to overlay.'. \
            format(img_olay_url)
        logger.error(msg=log)
        print(log)

    return img_olay_err


def shift_train_image(
        orig_path: str,
        shift_path: str,
        tf_dict: dict
) -> bool:
    """
    Shifts digit images and save shifted image

    :param orig_path: str
    :param shift_path: str
    :param tf_dict: dict

    :return shift_err: bool
    """
    shift_err = False

    if (tf_dict['tf_width'] % 2) == 0:
        tf_left = int(tf_dict['tf_width'] / 2)
        tf_right = int(tf_dict['tf_width'] / 2)
    else:
        tf_left = int((tf_dict['tf_width'] / 2) + 0.5)
        tf_right = int((tf_dict['tf_width'] / 2) - 0.5)

    try:
        for root, dirs, files in os.walk(top=orig_path):
            for filename in files:
                dest_path = os.path.join(shift_path, os.path.basename(root))
                im_file_orig = filename.split(sep='.')[0]
                digit = im_file_orig.split(sep='_')[0]
                img_path = os.path.join(orig_path, digit, os.path.basename(root), filename)
                im = Image.open(os.path.join(orig_path, img_path))
                if (im.size[0] % 2) == 0:
                    dig_ctr = int(im.size[0] / 2)
                else:
                    dig_ctr = int((im.size[0] / 2) + 0.5)

                if tf_dict['shift_en']:
                    for l_shift in range(0, tf_dict['shift']):
                        im_file = im_file_orig + '_sh' + str(l_shift) + '.jpg'
                        im_dims = [
                            ((tf_dict['full_width'] - tf_dict['tf_width'] - 1) - l_shift),
                            0,
                            ((tf_dict['full_width'] - 1) - l_shift),
                            im.size[1]
                        ]
                        im_crop = im.crop((
                            im_dims[0],  # upper left x
                            im_dims[1],  # upper left y
                            im_dims[2],  # bottom right x
                            im_dims[3]  # bottom right y
                        ))
                        im_crop.save(
                            os.path.join(dest_path, im_file),
                            "JPEG",
                            quality=100
                        )

                else:
                    im_file = im_file_orig + '_sh0.jpg'
                    im_dims = [
                        (dig_ctr - tf_left),
                        0,
                        (dig_ctr + tf_right),
                        im.size[1]
                    ]
                    im_crop = im.crop((
                        im_dims[0],  # upper left x
                        im_dims[1],  # upper left y
                        im_dims[2],  # bottom right x
                        im_dims[3]  # bottom right y
                    ))
                    im_crop.save(
                        os.path.join(dest_path, im_file),
                        "JPEG",
                        quality=100
                    )

    except Exception as exc:
        shift_err = True
        log = 'Failed to shift digits in path {0}.'.format(orig_path)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return shift_err


def save_test_image(
        orig_path: str,
        shift_path: str,
) -> bool:
    """
    Saves image

    :param orig_path: str
    :param shift_path: str

    :return shift_err: bool
    """
    shift_err = False

    try:
        for root, dirs, files in os.walk(top=orig_path):
            for filename in files:
                dest_path = os.path.join(shift_path, os.path.basename(root))
                im_file_orig = filename.split(sep='.')[0]
                digit = im_file_orig.split(sep='_')[0]
                img_path = os.path.join(orig_path, digit, os.path.basename(root), filename)
                im = Image.open(os.path.join(orig_path, img_path))

                im_file = im_file_orig + '_sh0.jpg'
                im.save(
                    os.path.join(dest_path, im_file),
                    "JPEG",
                    quality=100
                )

    except Exception as exc:
        shift_err = True
        log = 'Failed to shift digits in path {0}.'.format(orig_path)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return shift_err


def copy_image(
        img_orig_url: str,
        img_dest_path: str,
        img_dest_name: str,
        img_crop_dict: dict,
        img_dest_qual: int = 100,
        img_crop_en: bool = False
) -> (bool, str):
    """
    Crops and saves given image

    :param img_orig_url: str
    :param img_dest_path: str
    :param img_dest_name: str
    :param img_crop_dict: dict
    :param img_dest_qual: int
    :param img_crop_en: bool

    :return img_xmit_err: bool
    """
    img_copy_err = False

    img_dest_url = os.path.join(
        img_dest_path,
        img_dest_name
    )

    if os.path.isfile(path=img_orig_url):
        try:
            img_orig = Image.open(fp=img_orig_url)
            if img_crop_en:
                img_crop = img_orig.crop(box=(
                    img_crop_dict['ulx'],
                    img_crop_dict['uly'],
                    img_crop_dict['brx'],
                    img_crop_dict['bry']
                ))
                img_crop.save(
                    fp=img_dest_url,
                    format='jpeg',
                    optimize=True,
                    quality=img_dest_qual
                )
                img_crop.close()
            else:
                img_orig.save(
                    fp=img_dest_url,
                    format='jpeg',
                    optimize=True,
                    quality=img_dest_qual
                )
            img_orig.close()

            log = 'PIL successfully saved image {0}.'. \
                format(img_dest_url)
            logger.info(msg=log)

        except Exception as exc:
            img_copy_err = True
            log = 'PIL failed to save image {0}.'.format(img_dest_url)
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

    else:
        img_copy_err = True
        log = 'OS failed to locate image {0} to save.'. \
            format(img_orig_url)
        logger.error(msg=log)
        print(log)

    return img_copy_err


def remove_images(
        img_path: str
) -> None:
    """
    Deletes all images in given directory

    :param img_path: str
    """
    for root, dirs, imgs in os.walk(top=img_path):
        for img in imgs:
            os.unlink(path=os.path.join(img_path, img))


def preprocess_yolo(
        img_orig,
        target_size,
        gt_boxes=None
):
    preprocess_err = False
    img_padded = None
    ih, iw = target_size
    h, w, _ = img_orig.shape

    img_scale = min(iw / w, ih / h)
    nw, nh = int(img_scale * w), int(img_scale * h)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    try:
        img_resized = cv2.resize(
            src=img_orig,
            dsize=(nw, nh)
        )

        img_padded = np.full(
            shape=[ih, iw, 3],
            fill_value=128.0
        )

        img_padded[dh:nh + dh, dw:nw + dw, :] = img_resized
        img_padded = img_padded / 255.

    except Exception as exc:
        preprocess_err = True
        log = 'Failed to preprocess image {0}.'.format(img_orig)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    if gt_boxes is None:
        return preprocess_err, img_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * img_scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * img_scale + dh
        return preprocess_err, img_padded, gt_boxes


def draw_bbox_yolo(
        img_orig,
        bboxes,
        classes=None,
        show_label=True
):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    draw_bbox_err = False
    num_classes = len(classes)
    image_h, image_w, _ = img_orig.shape

    try:
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(
                bbox[:4],
                dtype=np.int32
            )
            font_scale = 0.5
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(
                img=img_orig,
                pt1=c1,
                pt2=c2,
                color=bbox_color,
                thickness=bbox_thick
            )

            if show_label:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTextSize(
                    text=bbox_mess,
                    fontFace=0,
                    fontScale=font_scale,
                    thickness=bbox_thick // 2
                )[0]
                cv2.rectangle(
                    img=img_orig,
                    pt1=c1,
                    pt2=(c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                    color=bbox_color,
                    thickness=-1
                )  # filled

                cv2.putText(
                    img=img_orig,
                    text=bbox_mess,
                    org=(c1[0], c1[1] - 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(0, 0, 0),
                    thickness=bbox_thick // 2,
                    lineType=cv2.LINE_AA
                )

    except Exception as exc:
        draw_bbox_err = True
        log = 'Failed to draw boundary boxes for image {0}.'.format(img_orig)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return draw_bbox_err, img_orig


def postprocess_boxes_yolo(
        img_orig_url,
        pred_bbox,
        org_img_shape,
        input_size,
        score_threshold
):
    postprocess_err = False
    bboxes = None
    best_bboxes = []

    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    try:
        # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate(
            [
                pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5
            ],
            axis=-1
        )

        # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = org_img_shape
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # # (3) clip some boxes those are out of range
        pred_coor = np.concatenate(
            [
                np.maximum(
                    pred_coor[:, :2],
                    [0, 0]
                ),
                np.minimum(
                    pred_coor[:, 2:],
                    [org_w - 1, org_h - 1]
                )
            ],
            axis=-1
        )
        invalid_mask = np.logical_or(
            (pred_coor[:, 0] > pred_coor[:, 2]),
            (pred_coor[:, 1] > pred_coor[:, 3])
        )
        pred_coor[invalid_mask] = 0

        # # (4) discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(
            pred_coor[:, 2:4] - pred_coor[:, 0:2],
            axis=-1
        ))
        scale_mask = np.logical_and(
            (valid_scale[0] < bboxes_scale),
            (bboxes_scale < valid_scale[1])
        )

        # # (5) discard some boxes with low scores
        classes = np.argmax(
            a=pred_prob,
            axis=-1
        )
        scores = pred_conf * pred_prob[
            np.arange(start=len(pred_coor)),
            classes
        ]
        score_mask = scores > score_threshold
        mask = np.logical_and(
            scale_mask,
            score_mask
        )
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
        bboxes = np.concatenate(
            [
                coors,
                scores[:, np.newaxis],
                classes[:, np.newaxis]
            ],
            axis=-1
        )

        classes_in_img = list(set(bboxes[:, 5]))

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            if len(cls_bboxes) > 0:
                max_ind = np.argmax(a=cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)

    except Exception as exc:
        postprocess_err = True
        log = 'Failed to postprocess boxes for image {0}.'.format(img_orig_url)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return postprocess_err, bboxes, best_bboxes
