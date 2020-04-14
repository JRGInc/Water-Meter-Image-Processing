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
                }
            }
        }
