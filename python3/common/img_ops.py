__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import cv2
import logging
import math
import os.path
from PIL import Image, ImageFont, ImageDraw

logfile = 'januswm'
logger = logging.getLogger(logfile)


def __find_y_edge(
        img_thresh,
        pixels,
        element: int,
        offset: int,
) -> int:
    """
    Determines vertical edge of digit window.

    :param img_thresh: opencv image
    :param pixels: list
    :param element: int
    :param offset: int

    :return edge: int
    """
    black_edge = True
    edge = None

    # Iterate through pixels
    for pix in pixels:

        # Ignore first pixels that are black, the digit window
        # will have white pixels between image edge and digit window
        # edge
        if img_thresh[pix][element] == 255:
            pass
        # Iterate over white pixels
        if img_thresh[pix][element] == 0:
            black_edge = False
            continue

        # If previous pixel was white and current pixel is black,
        # then the digit window edge is found, add offset
        if not black_edge and \
                img_thresh[pix][element] == 255:
            edge = pix + offset
            break

    return edge


def __find_x_edge(
        img_thresh,
        pixels,
        element: int,
        offset: int,
        left: bool,
) -> int:
    """
    Determines horizontal edge of digit window.

    :param img_thresh: opencv image
    :param pixels: list
    :param element: int
    :param offset: int
    :param left: bool

    :return edge: int
    """
    black_edge = True
    edge = None

    # Iterate through pixels
    for pix in pixels:

        # Ignore first pixels that are black, the digit window
        # will have white pixels between image edge and digit window
        # edge
        if img_thresh[element][pix] == 255:
            pass
        # Iterate over white pixels
        if img_thresh[element][pix] == 0:
            black_edge = False
            continue

        # If previous pixel was white and current pixel is black,
        # then the digit window edge is found, add offset
        if not black_edge and \
                img_thresh[element][pix] == 255:
            if left:
                edge = pix + offset
            else:
                edge = pix - offset
            break

    return edge


def find_angles(
        bbox_dict: dict
) -> tuple:
    """
    Determines angle of rotation of image from detected object
    bounding boxes

    :param bbox_dict: dict

    :return img_ang_err: bool
    :return img_ang_list: list
    """
    img_ang_err = False
    img_ang_list = []

    try:
        # Get center points of important bounding boxes and convert
        # points to integers
        img_ctr_x = int((bbox_dict['pivot'][1] + bbox_dict['pivot'][3]) / 2)
        img_ctr_y = int((bbox_dict['pivot'][2] + bbox_dict['pivot'][4]) / 2)
        screw_ctr_x = int((bbox_dict['screw'][1] + bbox_dict['screw'][3]) / 2)
        screw_ctr_y = int((bbox_dict['screw'][2] + bbox_dict['screw'][4]) / 2)
        digw_ctr_x = int((bbox_dict['digits'][1] + bbox_dict['digits'][3]) / 2)
        digw_ctr_y = int((bbox_dict['digits'][2] + bbox_dict['digits'][4]) / 2)

        # Calculate approximate angle for center of digit window bounding box
        digw_ang = math.degrees(
            math.atan2(
                digw_ctr_y - img_ctr_y,
                digw_ctr_x - img_ctr_x
            )
        )
        digw_ang = round(digw_ang, 2)
        if digw_ang < 0:
            digw_ang = 360 + digw_ang

        # Calculate approximate angle for center of screw bounding box
        screw_ang = math.degrees(
            math.atan2(
                screw_ctr_y - img_ctr_y,
                screw_ctr_x - img_ctr_x
            )
        )
        screw_ang = round(screw_ang, 2)
        if screw_ang < 0:
            screw_ang = 360 + screw_ang

        # Determine angle difference to get image orientation, and calculate
        # gross rotation angle
        diff_ang = digw_ang - screw_ang
        grot_ang = 0
        if (-105 <= diff_ang <= -75) or (255 <= diff_ang <= 285):
            grot_ang = 180 + screw_ang
        elif (75 <= diff_ang <= 105) or (-285 <= diff_ang <= -255):
            grot_ang = screw_ang

        img_ang_list = [img_ctr_x, img_ctr_y, grot_ang, diff_ang, screw_ang, digw_ang]

    except Exception as exc:
        img_ang_err = True
        log = 'OpenCV failed to determine angles.'
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return img_ang_err, img_ang_list


def rotate(
        img_orig,
        img_save: bool,
        img_grotd_url: str,
        img_frotd_url: str,
        img_orig_shape: str,
        img_ang_list: list
) -> tuple:
    """
    Grossly rotates given image based on bounding box angles, afterwards
    performs fine rotation based on top edge of digit window

    :param img_orig: opencv image
    :param img_save: bool
    :param img_grotd_url: str
    :param img_frotd_url: str
    :param img_orig_shape: str
    :param img_ang_list: list

    :return img_rotd: opencv image
    :return img_rotd_err: bool
    """
    img_frotd = None
    img_rotd_err = False
    img_x_offset = None
    img_y_offset = None

    try:
        # Use pivot center and gross rotation angle, get rotation matrix
        # and grossly rotate image
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
            dsize=(
                img_orig_shape[1],
                img_orig_shape[0]
            )
        )
        if img_save:
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

        # Set crop parameters, and crop extraneous features from image
        # so that they do not interfere with determining fine rotation angle
        # from upper edge of digit window.
        #
        # Horizontal offset is used to ensure that corner deformations do
        # not affect the fine rotation angle determined from the upper edge
        #
        # Vertical offset is used to ensure that left and right vertical
        # edges can be found when digit window is slightly angle.
        img_crop_dict = None
        if img_orig_shape[0] == 1536:
            img_crop_dict = {
                'ulx': img_ang_list[0] - 240,
                'uly': img_ang_list[1] + 90,
                'brx': img_ang_list[0] + 240,
                'bry': img_ang_list[1] + 205
            }
            img_x_offset = 10
            img_y_offset = 10

        elif img_orig_shape[0] == 2464:
            img_crop_dict = {
                'ulx': img_ang_list[0] - 390,
                'uly': img_ang_list[1] + 140,
                'brx': img_ang_list[0] + 390,
                'bry': img_ang_list[1] + 320
            }
            img_x_offset = 10
            img_y_offset = 20

        img_rect = img_grotd[
            img_crop_dict['uly']:img_crop_dict['bry'],
            img_crop_dict['ulx']:img_crop_dict['brx']
        ]
        img_rect_ctr_x = int(img_rect.shape[1] / 2)
        img_rect_ctr_y = int(img_rect.shape[0] / 2)

        # Convert cropped image to black and white inverted
        # image in order to find upper edge of digit window
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

        # Iterate over y-pixels from top to bottom to find upper
        # edge at image width center point, add vertical offset.
        y_pixels = iter(range(0, img_rect.shape[0]))
        upper = __find_y_edge(
            img_thresh=img_thresh,
            pixels=y_pixels,
            element=img_rect_ctr_x,
            offset=img_y_offset
        )

        # Iterate over x-pixels from left to right at upper edge plus
        # offset to find left edge of digit window, add horizontal offset
        x_pixels = iter(range(0, img_rect.shape[1]))
        left = __find_x_edge(
            img_thresh=img_thresh,
            pixels=x_pixels,
            element=upper,
            offset=img_x_offset,
            left=True
        )

        # Iterate over x-pixels from right to left at upper edge plus
        # offset to find left edge of digit window, subtract horizontal offset
        x_pixels = iter(range((img_rect.shape[1] - 1), 0, -1))
        right = __find_x_edge(
            img_thresh=img_thresh,
            pixels=x_pixels,
            element=upper,
            offset=img_x_offset,
            left=False
        )

        # Iterate over y-pixels from top to bottom at left edge plus offset
        # to find upper edge at image left plus offset.
        y_pixels = iter(range(0, img_rect.shape[0]))
        l_upper = __find_y_edge(
            img_thresh=img_thresh,
            pixels=y_pixels,
            element=left,
            offset=0
        )

        # Iterate over y-pixels from top to bottom at right edge minus offset
        # to find upper edge at image left plus offset.
        y_pixels = iter(range(0, img_rect.shape[0]))
        r_upper = __find_y_edge(
            img_thresh=img_thresh,
            pixels=y_pixels,
            element=right,
            offset=0
        )

        # Use left and right upper edge points to find fine rotation angle
        img_rect_ang = math.degrees(
            math.atan2(
                r_upper - l_upper,
                right - left
            )
        )
        img_rect_ang = round(img_rect_ang, 2)
        img_frotd_ctr = [
            img_rect_ctr_x,
            img_rect_ctr_y
        ]

        # Use rectangle center and fine rotation angle, get rotation matrix
        # and finely rotate image
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
            dsize=(
                img_rect.shape[1],
                img_rect.shape[0]
            )
        )
        if img_save:
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
        log = 'Failed to rotate image.'
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return img_rotd_err, img_frotd


def crop_rect(
        img_rotd,
        img_save: bool,
        img_rect_url: str,
        img_digw_url: str
) -> tuple:
    """
    Crops digit window to digit window edges

    :param img_rotd: opencv image
    :param img_save: bool
    :param img_rect_url: str
    :param img_digw_url: str

    :return img_digw: opencv image
    :return img_rect_err: bool
    """
    img_digw = None
    img_rect_err = False

    try:
        # Determine horizontal center point of rotated image
        img_rotd_ctr_x = int(img_rotd.shape[1] / 2)

        # Convert cropped image to black and white inverted
        # image in order to find upper edge of digit window
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

        lx_offset = 3
        rx_offset = -23
        uy_offset = 3
        ly_offset = -1

        y_pixels = iter(range((img_rotd.shape[0] - 1), 0, -1))
        lower = __find_y_edge(
            img_thresh=img_thresh,
            pixels=y_pixels,
            element=img_rotd_ctr_x,
            offset=0
        )

        y_pixels = iter(range(0, img_rotd.shape[0]))
        upper = __find_y_edge(
            img_thresh=img_thresh,
            pixels=y_pixels,
            element=img_rotd_ctr_x,
            offset=0
        )

        x_pixels = iter(range(0, img_rotd.shape[1]))
        left = __find_x_edge(
            img_thresh=img_thresh,
            pixels=x_pixels,
            element=(upper + uy_offset),
            offset=0,
            left=True
        )

        y_half = int((lower + upper)/2)
        x_pixels = iter(range((img_rotd.shape[1] - 1), 0, -1))
        right = __find_x_edge(
            img_thresh=img_thresh,
            pixels=x_pixels,
            element=y_half,
            offset=0,
            left=False
        )

        log = 'Calculated raw digit window edges at:      ' + \
            '{0}, {1}, {2}, {3} (upper, lower, left, right).'.\
            format(upper, lower, left, right)
        logger.info(msg=log)
        print(log)

        ulx = left + lx_offset
        uly = upper + uy_offset
        brx = right + rx_offset
        bry = lower + ly_offset

        # Make image width evenly divisible by 6 to facilitate
        # cropping the digit window evenly
        length_rem = (brx - ulx) % 6
        if (length_rem >= 1) and (length_rem < 4):
            brx -= length_rem
        elif (length_rem >= 4) and (length_rem < 6):
            brx += 6 - length_rem

        img_digw = img_rotd[
            uly:bry,
            ulx:brx
        ]

        log = 'Calculated adjusted digit window edges at: ' + \
            '{0}, {1}, {2}, {3} (upper, lower, left, right).'. \
            format(uly, bry, ulx, brx)
        logger.info(msg=log)
        print(log)

        if img_save:
            cv2.line(
                img=img_rotd,
                pt1=(0, (uly - 1)),
                pt2=(img_rotd.shape[1], (uly - 1)),
                color=(0, 255, 0),
                thickness=1
            )
            cv2.line(
                img=img_rotd,
                pt1=(0, bry),
                pt2=(img_rotd.shape[1], bry),
                color=(0, 255, 0),
                thickness=1
            )
            cv2.line(
                img=img_rotd,
                pt1=((ulx - 1), 0),
                pt2=((ulx - 1), img_rotd.shape[0]),
                color=(0, 255, 0),
                thickness=1
            )
            cv2.line(
                img=img_rotd,
                pt1=(brx, 0),
                pt2=(brx, img_rotd.shape[0]),
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

            cv2.imwrite(
                filename=img_digw_url,
                img=img_digw,
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

    return img_rect_err, img_digw


def crop_digits(
        img_digw,
        img_save: bool,
        img_inv_url: str,
        img_path_dict: dict
) -> (bool, list):
    """
    Crops and saves given image as separate digits

    :param img_digw: opencv image
    :param img_save: bool
    :param img_inv_url: str
    :param img_path_dict: dict

    :return img_digs_err: bool
    :return img_digs: list
    """
    img_digs = [None, None, None, None, None, None]
    img_digs_err = False
    img_upper = 0

    try:
        # Calculate size of individual digits
        img_lower = img_digw.shape[0] - 1
        dig_w = int(img_digw.shape[1] / 6)
        dig_w_ctr = int(img_digw.shape[1] / 2)

        log = 'Raw digit width is {0} pixels and center is {1} pixels.'. \
            format(dig_w, dig_w_ctr)
        logger.info(msg=log)
        print(log)

        # Convert cropped image to black and white inverted
        # image in order to find upper edge of digit window
        img_gray = cv2.cvtColor(
            src=img_digw,
            code=cv2.COLOR_BGR2GRAY
        )
        thresh, img_thresh = cv2.threshold(
            src=img_gray,
            thresh=120,
            maxval=255,
            type=cv2.THRESH_BINARY_INV
        )
        img_edge = cv2.Canny(
            image=img_thresh,
            threshold1=30,
            threshold2=200
        )

        if img_save:
            cv2.imwrite(
                filename=img_inv_url,
                img=img_thresh,
                params=[
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    100
                ]
            )

        # Iterate through all six digits
        for digit in range(0, 6):

            # Calculate horizontal start point and end point
            # for each digit
            start_x = digit * dig_w
            end_x = start_x + dig_w

            img_dig_gray = img_gray[
                0:img_digw.shape[0],
                start_x:end_x
            ].copy()

            # Must find contour of greatest width and set tensor window accordingly.
            # Assume widest contour is the shape of the number on the digit.
            # This may include glare artifacts that are wider than the digit.
            contours, hierarchy = cv2.findContours(
                image=img_edge[
                    0:img_digw.shape[0],
                    start_x:end_x
                ],
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE
            )

            # Iterate through list of contours
            cnt_w = 0
            cnt_l = 0
            for contour in range(0, len(contours)):

                # Set bounding rectangle on contour
                x, y, w, h = cv2.boundingRect(contours[contour])

                # Keep left horizontal edge and width
                # of largest contour
                if (w > 10) and (w > cnt_w):
                    cnt_w = w
                    cnt_l = x

            # Determine horizontal center and
            # right pixels of widest contour
            cnt_ctr = cnt_l + int(cnt_w / 2)
            cnt_r = cnt_l + cnt_w

            # Add horizontal buffers to left and right limits
            # to create new number limits
            left = cnt_l - 10
            right = cnt_r + 10

            # If contour is erroneously left-shifted beyond
            # digit left edge, set number limit to 0
            if left < 0:
                left = 0

            # If contour is erroneously right-shifted beyond
            # digit right edge, set number limit to right edge
            elif right >= dig_w:
                right = dig_w - 1

            # Create digit image from digit window in gray-scale
            img_digs[digit] = img_dig_gray[
                img_upper:img_lower,
                left:right
            ]

            log = 'Digit {0} tensor flow adjusted horizontal boundaries '.\
                format(digit) + \
                'are: {0}, {1}, and {2} (left, center, right).'.\
                format(left, cnt_ctr, right)
            logger.info(msg=log)
            print(log)

            log = 'Digit {0} tensor flow raw horizontal boundaries '. \
                format(digit) + \
                'are:      {0}, {1}, and {2} (left, center, right).'. \
                format(cnt_l, cnt_ctr, cnt_r)
            logger.info(msg=log)
            print(log)

            if img_save:
                # Need to make copy of image in memory to prevent
                # OpenCV operations from changing original image
                img_dig_orig = img_digw[
                    0:img_digw.shape[0],
                    start_x:end_x
                ].copy()

                img_dig_cnt = cv2.drawContours(
                    image=img_dig_orig,
                    contours=contours,
                    contourIdx=-1,
                    color=(255, 0, 255),
                    thickness=1
                )

                cv2.line(
                    img=img_dig_cnt,
                    pt1=(0, img_upper),
                    pt2=(img_digw.shape[1], img_upper),
                    color=(0, 255, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(0, img_lower),
                    pt2=(img_digw.shape[1], img_lower),
                    color=(0, 255, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(cnt_l, 0),
                    pt2=(cnt_l, img_digw.shape[0]),
                    color=(0, 255, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(cnt_ctr, 0),
                    pt2=(cnt_ctr, img_digw.shape[0]),
                    color=(0, 0, 255),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(cnt_r, 0),
                    pt2=(cnt_r, img_digw.shape[0]),
                    color=(0, 255, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(left, 0),
                    pt2=(left, img_digw.shape[0]),
                    color=(255, 0, 0),
                    thickness=1
                )
                cv2.line(
                    img=img_dig_cnt,
                    pt1=(right, 0),
                    pt2=(right, img_digw.shape[0]),
                    color=(255, 0, 0),
                    thickness=1
                )

                img_cont_url = os.path.join(
                    img_path_dict['cont'],
                    'cont' + '_d' + str(digit) + os.path.basename(img_inv_url)[3::]
                )
                cv2.imwrite(
                    filename=img_cont_url,
                    img=img_dig_cnt,
                    params=[
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        100
                    ]
                )

                img_dig_url = os.path.join(
                    img_path_dict['digs'],
                    'digs' + '_d' + str(digit) + os.path.basename(img_inv_url)[3::]
                )

                cv2.imwrite(
                    filename=img_dig_url,
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

    return img_digs_err, img_digs


def overlay(
        img_orig_shape: str,
        img_digw,
        img_olay_url: str,
        img_olay_text: str
) -> bool:
    """
    Adds meter value banner to bottom of given image and saves

    :param img_orig_shape: str
    :param img_digw: opencv image
    :param img_olay_url: str
    :param img_olay_text: str

    :return img_olay_err: bool
    """
    img_olay_err = False

    try:
        # Convert OpenCV image to PIL image
        img_digw = cv2.cvtColor(
            src=img_digw,
            code=cv2.COLOR_BGR2RGB
        )
        img_digw_pil = Image.fromarray(obj=img_digw)
        brx, bry = img_digw_pil.size

        # Create new PIL image with vertical size greater than
        # vertical size of digit window image
        img_olay = Image.new(
            mode='RGB',
            size=(brx, (bry + 30)),
            color=(0, 0, 0)
        )

        # Paste digit window image into new PIL image at top, save,
        # then close digit window image
        img_olay.paste(
            im=img_digw_pil,
            box=(0, 0)
        )
        img_olay.save(
            fp=img_olay_url,
            format='jpeg',
            optimize=True,
            quality=100
        )
        img_digw_pil.close()

        # Setup image draw object in order to add overlay text
        # to bottom of image
        img_olay_draw = ImageDraw.Draw(im=img_olay)

        if img_orig_shape[0] == 2464:
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
        elif img_orig_shape[0] == 1536:
            img_olay_font = ImageFont.truetype(
                font="DejaVuSans.ttf",
                size=11
            )
            img_olay_draw.text(
                xy=(10, (bry + 6)),
                text=img_olay_text,
                fill=(255, 255, 0, 255),
                font=img_olay_font
            )
        else:
            img_olay_font = ImageFont.truetype(
                font="DejaVuSans.ttf",
                size=11
            )
            img_olay_draw.text(
                xy=(10, (bry + 6)),
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
        print(log)

    except Exception as exc:
        img_olay_err = True
        log = 'PIL failed to overlay image {0}.'. \
            format(img_olay_url)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return img_olay_err
