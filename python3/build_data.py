import csv
import cv2
import os
from common import file_ops

if __name__ == '__main__':
    image_path = '/mnt/data/01--FILES/02--WATER_METER/04--MACHINE/02--DETECTION/04--DETECTION/images/WM_valid/'
    tboxes_path = '/mnt/data/01--FILES/02--WATER_METER/04--MACHINE/02--DETECTION/04--DETECTION/images/WM_train_boxes/'
    vboxes_path = '/mnt/data/01--FILES/02--WATER_METER/04--MACHINE/02--DETECTION/04--DETECTION/images/WM_valid_boxes/'
    data_path = '/mnt/data/01--FILES/02--WATER_METER/04--MACHINE/02--DETECTION/04--DETECTION/images/'
    data_url = os.path.join(
        data_path,
        'WM_valid.txt'
    )
    data_out = [None]

    # for image in sorted(os.listdir(image_path)):
    #     image_name, image_ext = os.path.splitext(image)
    #     if image_ext == '.jpg':
    #         images_url = os.path.join(image_path, (image_name + '.txt'))
    #         boxes_url = os.path.join(boxes_path, (image_name + '.txt'))
    #         print(images_url)
    #         data_in = file_ops.f_request(
    #             file_cmd='data_line_read',
    #             file_name=images_url
    #         )
    #         for row in data_in:
    #             # row = row.rstrip(os.linesep)
    #             print(row)
    #             file_ops.f_request(
    #                 file_cmd='file_line_append',
    #                 file_name=boxes_url,
    #                 data_file_in=[row]
    #             )

    img_h = None
    img_w = None

    data_url_open = open(
        file=data_url,
        mode='a',
        encoding='utf-8'
    )
    for image in sorted(os.listdir(image_path)):
        image_url = os.path.join(image_path, image)
        image_name, image_ext = os.path.splitext(image)
        if image_ext == '.jpg':
            print(image_url)
            data_out[0] = image_url + ' '
            img = cv2.imread(filename=image_url)
            img_h = img.shape[0]
            img_w = img.shape[1]

            boxes_url = os.path.join(vboxes_path, (image_name + '.txt'))
            print(boxes_url)
            data_in = file_ops.f_request(
                file_cmd='data_line_read',
                file_name=boxes_url
            )
            for row in data_in:
                row = row.rstrip(os.linesep)
                fields = row.split(sep=' ')
                XCtr = int(float(fields[1]) * img_w)
                YCtr = int(float(fields[2]) * img_h)
                XOffset = int((float(fields[3]) * img_w) / 2)
                YOffset = int((float(fields[4]) * img_h) / 2)
                XMin = str(XCtr - XOffset)
                YMin = str(YCtr - YOffset)
                XMax = str(XCtr + XOffset)
                YMax = str(YCtr + YOffset)
                data_out[0] += XMin + ',' +\
                    YMin + ',' +\
                    XMax + ',' +\
                    YMax + ',' +\
                    fields[0] + ' '

            data_out[0] += '\n'
            print(data_out)

            file_ops.f_request(
                file_cmd='file_line_append',
                file_name=data_url,
                data_file_in=data_out
            )

            data_out = [None]
    data_url_open.close()

    # for image in sorted(os.listdir(image_path)):
    #     image_name, image_ext = os.path.splitext(image)
    #     if image_ext == '.jpg':
    #         images_url = os.path.join(image_path, (image_name + '.txt'))
    #         tboxes_url = os.path.join(tboxes_path, (image_name + '.txt'))
    #         vboxes_url = os.path.join(vboxes_path, (image_name + '.txt'))
    #
    #         file_ops.copy_file(
    #             data_orig_url=tboxes_url,
    #             data_dest_url=vboxes_url
    #         )
