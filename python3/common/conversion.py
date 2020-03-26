__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import signal
from common import os_cmd

logfile = 'januswm-train'
logger = logging.getLogger(logfile)


def h264_to_jpg(
    vid_h264_url: str,
    vid_fps_int: int,
    img_orig_url: str
) -> bool:
    """
    Captures and saves original image

    :param vid_h264_url: str
    :param vid_fps_int: int
    :param img_orig_url: str

    :return img_orig_err: bool
    """
    conv_err = False
    cmd_err_count = 0
    timeout = 30

    def timeout_handler(
        signum,
        frame
    ):
        raise Exception('Timeout occurred for h264 to jpg conversion.')

    signal.signal(
        signal.SIGALRM,
        timeout_handler
    )

    while cmd_err_count < 2:

        signal.alarm(timeout)
        try:

            cmd_str = 'ffmpeg ' +\
                '-i ' + str(vid_h264_url) + ' ' +\
                '-r ' + str(vid_fps_int) + ' ' +\
                '-s 1920x1080 ' +\
                '-f image2 ' +\
                '-q:v 1 ' + \
                str(img_orig_url)
            cmd_err, rtn_code, std_out = os_cmd.os_cmd(
                cmd_str=cmd_str
            )

            if cmd_err and (rtn_code == 0):
                break

            else:
                conv_err = True
                cmd_err_count += 1
                signal.alarm(0)
                break

        except Exception as exc:
            conv_err = True
            cmd_err_count += 1
            log = 'Timeout took place at {0} seconds'. \
                format(timeout)
            logger.error(msg=exc)
            logger.error(msg=log)
            print(exc)
            print(log)

        finally:
            signal.alarm(0)

    return conv_err
