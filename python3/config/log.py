__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import os

# Log paths and files
LOGPATH = os.path.normpath(
    os.getenv(
        'JANUSWM_CORE_LOG_PATH',
        '/var/log/JanusWM/'
    )
)
januswm = os.path.join(LOGPATH, 'januswm')                      # Log file
januswm_capture = os.path.join(LOGPATH, 'januswm-capture')      # Log file
januswm_video = os.path.join(LOGPATH, 'januswm-video')          # Log file
januswm_train = os.path.join(LOGPATH, 'januswm-train')          # Log file
januswm_test = os.path.join(LOGPATH, 'januswm-test')          # Log file
januswm_transmit = os.path.join(LOGPATH, 'januswm-transmit')    # Log file


class LogCfg(object):
    def __init__(
        self
    ) -> None:
        """
        Instantiates logging object and sets log configuration
        """
        self.config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(levelname)s %(message)s',
                },
                'verbose': {
                    'format': " * %(asctime)s * %(levelname)s: " +
                              "<function '%(funcName)s' from '%(filename)s'>: %(message)s",
                },
            },
            'loggers': {
                'januswm': {
                    'handlers': ['januswm'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'januswm-capture': {
                    'handlers': ['januswm-capture'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'januswm-video': {
                    'handlers': ['januswm-video'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'januswm-train': {
                    'handlers': ['januswm-train'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'januswm-test': {
                    'handlers': ['januswm-test'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'januswm-transmit': {
                    'handlers': ['januswm-transmit'],
                    'propagate': False,
                    'level': 'INFO',
                }
            },
            'handlers': {
                'januswm': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': januswm,
                    'maxBytes': 4096000,
                    'backupCount': 100,
                },
                'januswm-capture': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': januswm_capture,
                    'maxBytes': 4096000,
                    'backupCount': 100,
                },
                'januswm-video': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': januswm_video,
                    'maxBytes': 4096000,
                    'backupCount': 100,
                },
                'januswm-train': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': januswm_train,
                    'maxBytes': 4096000,
                    'backupCount': 100,
                },
                'januswm-test': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': januswm_test,
                    'maxBytes': 4096000,
                    'backupCount': 100,
                },
                'januswm-transmit': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': januswm_transmit,
                    'maxBytes': 4096000,
                    'backupCount': 100,
                }
            }
        }
