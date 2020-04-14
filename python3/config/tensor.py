__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import configparser
import logging
import os

logfile = 'januswm'
logger = logging.getLogger(logfile)


class TensorCfg(object):
    """
    Class attributes of configuration settings for TensorFlow operations
    """
    def __init__(
        self,
        core_cfg: any
    ) -> None:
        """
        Sets object properties either directly or via ConfigParser from ini file

        :param core_cfg: any
        """

        # Setup core paths
        core_path_dict = core_cfg.get(attrib='core_path_dict')
        self.tensor_cfg_url = os.path.join(
            core_path_dict['cfg'],
            'tensor.ini'
        )

        # Setup config parser to open and read *.ini files
        self.config = configparser.ConfigParser()
        self.config.read_file(f=open(self.tensor_cfg_url))

        # Dimensions for individual digits
        self.dig_dict = {
            'full_width': self.config.getint(
                'Digit',
                'full_width'
            ),
            'dig_width': self.config.getint(
                'Digit',
                'dig_width'
            ),
            'shadow': self.config.getint(
                'Digit',
                'shadow'
            ),
            'height': self.config.getint(
                'Digit',
                'height'
            ),
            'shift_en': self.config.getboolean(
                'Digit',
                'shift_en'
            ),
            'shift': self.config.getint(
                'Digit',
                'shift'
            )
        }

        # Global Inception object detection settings
        self.incept_dict = {
            'mdl': os.path.join(
                core_path_dict['mdls'],
                'inception_v4/pb/'
            ),
            'wgts': os.path.join(
                core_path_dict['wgts'],
                'inception_v4/final/final.ckpt'
            ),
            'batch_size': self.config.getint(
                'Inception',
                'batch_size'
            ),
            'img_tgt_width': self.config.getint(
                'Inception',
                'img_tgt_width'
            ),
            'img_tgt_height': self.config.getint(
                'Inception',
                'img_tgt_height'
            ),
            'nbr_classes': self.config.getint(
                'Inception',
                'nbr_classes'
            ),
            'nbr_channels': self.config.getint(
                'Inception',
                'nbr_channels'
            ),
            'patience': self.config.getint(
                'Inception',
                'patience'
            ),
            'epochs': self.config.getint(
                'Inception',
                'epochs'
            ),
            'format': self.config.get(
                'Inception',
                'format'
            ),
            'confidence': self.config.getfloat(
                'Inception',
                'confidence'
            )
        }

        # Global YOLO object detection settings
        self.yolo_dict = {
            'classes': os.path.join(
                core_path_dict['cfg'],
                'yolo_classes.txt'
            ),
            'anchors': os.path.join(
                core_path_dict['cfg'],
                'yolo_v3_anchors.txt'
            ),
            'mdl': os.path.join(
                core_path_dict['mdls'],
                'yolo_v3/'
            ),
            'wgts': os.path.join(
                core_path_dict['wgts'],
                'yolo_v3/yolo_v3'
            ),
            'rslts': os.path.join(
                core_path_dict['rslts'],
                'test/'
            ),
            'strides': [8, 16, 32],
            'anchor_per_scale': self.config.getint(
                'Yolo',
                'anchor_per_scale'
            ),
            'iou_loss_thresh': self.config.getfloat(
                'Yolo',
                'iou_loss_thresh'
            ),
            'batch_size': self.config.getint(
                'Yolo',
                'batch_size'
            ),
            'input_size': self.config.getint(
                'Yolo',
                'input_size'
            ),
            'data_aug': self.config.getboolean(
                'Yolo',
                'data_aug'
            ),
            'score_thresh': self.config.getfloat(
                'Yolo',
                'score_thresh'
            ),
            'iou_thresh': self.config.getfloat(
                'Yolo',
                'iou_thresh'
            )
        }

    def get(
        self,
        attrib: str
    ) -> any:
        """
        Gets configuration attributes

        :param attrib: str

        :return: any
        """
        if attrib == 'dig_dict':
            return self.dig_dict
        elif attrib == 'incept_dict':
            return self.incept_dict
        elif attrib == 'yolo_dict':
            return self.yolo_dict

    def set(
        self,
        section: str,
        attrib: str,
        value: str
    ) -> (bool, str):
        """
        Sets configuration attributes by updating ini file

        :param section: str
        :param attrib: str
        :param value: str

        :return: set_err: bool
        """
        set_err = False
        valid_section = True
        valid_option = True
        valid_value = True
        log = ''

        if section == 'Digit':
            opt_type = 'int'

            if attrib == 'full_width':
                pass

            elif attrib == 'dig_width':
                pass

            elif attrib == 'shadow':
                pass

            elif attrib == 'height':
                pass

            elif attrib == 'shift_en':
                opt_type = 'bool'
                pass

            elif attrib == 'shift':
                pass

            else:
                valid_option = False

            if valid_option and (opt_type == 'int'):
                try:
                    if int(value) < 1:
                        log = 'Attribute value {0} is less than 1: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                except ValueError:
                    log = 'Attribute value {0} is not an integer: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            if valid_option and (opt_type == 'bool'):
                if not (value == 'True') and not (value == 'False'):
                    log = 'Attribute value {0} is not a boolean: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

        elif section == 'Inception':
            opt_type = 'int'

            if attrib == 'batch_size':
                pass

            elif attrib == 'img_tgt_width':
                pass

            elif attrib == 'img_tgt_height':
                pass

            elif attrib == 'nbr_classes':
                pass

            elif attrib == 'patience':
                pass

            elif attrib == 'epochs':
                pass

            elif attrib == 'format':
                opt_type = 'str'
                pass

            elif attrib == 'confidence':
                opt_type = 'float'

            else:
                valid_option = False

            if valid_option and (opt_type == 'int'):
                try:
                    if int(value) < 1:
                        log = 'Attribute value {0} is less than 1: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                except ValueError:
                    log = 'Attribute value {0} is not an integer: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            if valid_option and (opt_type == 'float'):
                try:
                    if (float(value) < 0.00) and (float(value) > 1.00):
                        log = 'Attribute value {0} is not within accepted numerical limits.'. \
                            format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False
                except ValueError:
                    log = 'Attribute {0} value {1} is not a float.'. \
                        format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

        elif section == 'Yolo':
            opt_type = 'int'

            if attrib == 'anchor_per_scale':
                pass

            elif attrib == 'iou_loss_thresh':
                opt_type = 'float'
                pass

            elif attrib == 'batch_size':
                pass

            elif attrib == 'input_size':
                pass

            elif attrib == 'data_aug':
                opt_type = 'bool'
                pass

            elif attrib == 'score_thresh':
                opt_type = 'float'
                pass

            elif attrib == 'iou_thresh':
                opt_type = 'float'

            else:
                valid_option = False

            if valid_option and (opt_type == 'int'):
                try:
                    if int(value) < 1:
                        log = 'Attribute value {0} is less than 1: {1}.'.format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False

                except ValueError:
                    log = 'Attribute value {0} is not an integer: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            if valid_option and (opt_type == 'float'):
                try:
                    if (float(value) < 0.00) and (float(value) > 1.00):
                        log = 'Attribute value {0} is not within accepted numerical limits.'. \
                            format(attrib, value)
                        logger.error(msg=log)
                        log1 = 'Retaining previous value.'.format(attrib)
                        logger.warning(msg=log1)
                        valid_value = False
                except ValueError:
                    log = 'Attribute {0} value {1} is not a float.'. \
                        format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

            if valid_option and (opt_type == 'bool'):
                if not (value == 'True') and not (value == 'False'):
                    log = 'Attribute value {0} is not a boolean: {1}.'.format(attrib, value)
                    logger.error(msg=log)
                    log1 = 'Retaining previous value.'.format(attrib)
                    logger.warning(msg=log1)
                    valid_value = False

        else:
            valid_section = False
            valid_option = False

        if valid_section:
            if valid_option:
                if valid_value:
                    try:
                        self.config.set(
                            section=section,
                            option=attrib,
                            value=value
                        )
                        self.config.write(
                            fp=open(
                                file=self.tensor_cfg_url,
                                mode='w',
                                encoding='utf-8'
                            ),
                            space_around_delimiters=True
                        )
                        log = 'ConfigParser successfully set and wrote options to file.'
                        logger.info(msg=log)
                    except Exception as exc:
                        log = 'ConfigParser failed to set and write options to file.'
                        logger.error(msg=log)
                        logger.error(msg=exc)
                        print(log)
                        print(exc)

            else:
                set_err = True
                log = 'ConfigParser failed to set and write options to file, invalid option.'
                logger.error(msg=log)

        else:
            set_err = True
            log = 'ConfigParser failed to set and write options to file, invalid section.'
            logger.error(msg=log)

        return set_err, log
