__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import os

logfile = 'januswm'
logger = logging.getLogger(logfile)


class CoreCfg(object):
    """
    Class attributes of configuration settings for capture operations
    """

    def __init__(
            self
    ) -> None:
        """
        Sets object properties directly
        """
        base_dir = os.path.dirname('/opt/Janus/WM/')
        program_dir = '2020_METER_PROCESSING/'
        self.program_dir = os.path.join(
            base_dir,
            program_dir
        )

        # Define core paths
        core_dirs_dict = {
            'cfg': 'config/',
            'imgs': 'images/',
            'logs': 'logs/',
            'mdls': 'models/',
            'py3': 'python3/',
            'rslts': 'results/',
            'wgts': 'weights/'
        }

        # Define core paths
        self.core_path_dict = {
            'cfg': os.path.join(
                self.program_dir,
                core_dirs_dict['cfg']
            ),
            'imgs': os.path.join(
                self.program_dir,
                core_dirs_dict['imgs']
            ),
            'logs': os.path.join(
                self.program_dir,
                core_dirs_dict['logs']
            ),
            'mdls': os.path.join(
                self.program_dir,
                core_dirs_dict['mdls']
            ),
            'py3': os.path.join(
                self.program_dir,
                core_dirs_dict['py3']
            ),
            'rslts': os.path.join(
                self.program_dir,
                core_dirs_dict['rslts']
            ),
            'wgts': os.path.join(
                self.program_dir,
                core_dirs_dict['wgts']
            )
        }

        # Define image paths
        img_dirs_dict = {
            'orig': '01--original/',
            'bbox': '02--bboxes/',
            'grotd': '03--grotated/',
            'frotd': '04--frotated/',
            'rect': '05--rectangled/',
            'digw': '06--windowed/',
            'inv': '07--inverted',
            'cont': '08--contoured/',
            'digs': '09--digits/',
            'pred': '10--prediction/',
            'olay': '11--overlaid/'
        }

        # Define full urls for files in /images path
        self.img_path_dict = {
            'orig': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['orig']
            ),
            'bbox': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['bbox']
            ),
            'grotd': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['grotd']
            ),
            'frotd': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['frotd']
            ),
            'rect': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['rect']
            ),
            'digw': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['digw']
            ),
            'inv': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['inv']
            ),
            'cont': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['cont']
            ),
            'digs': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['digs']
            ),
            'pred': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['pred']
            ),
            'olay': os.path.join(
                self.core_path_dict['imgs'],
                img_dirs_dict['olay']
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
        if attrib == 'program_dir':
            return self.program_dir
        elif attrib == 'core_path_dict':
            return self.core_path_dict
        elif attrib == 'img_path_dict':
            return self.img_path_dict
