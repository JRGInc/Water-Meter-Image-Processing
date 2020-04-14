__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import subprocess

logfile = 'januswm'
logger = logging.getLogger(logfile)


def os_cmd(
    cmd_str: str,
) -> tuple:
    """
    Executes operating system command

    :param cmd_str: str

    :return cmd_err: bool
    :return cmd_std_out: any
    """
    cmd_err = False

    logger.info(cmd_str)
    print(cmd_str)

    # Feed command string into OS subprocess
    process = subprocess.Popen(
        cmd_str.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    std_out, std_err = process.communicate()

    # Decode command results
    std_out = std_out.decode('utf-8')
    std_err = std_err.decode('utf-8')
    rtn_code = process.returncode

    # Parse command results for logging
    if std_out == '':
        log = 'No reported output from command.'
        logger.info(msg=log)
        print(log)
    else:
        logger.info(msg=std_out)

    if std_err == '':
        log = 'No reported error from command.'
        logger.info(msg=log)
        print(log)
    else:
        cmd_err = True
        log = 'Error reported by command: {0}.'.format(std_err)
        logger.error(msg=log)
        print(std_err)

    if rtn_code == 0:
        log = 'Successfully executed command and returned code: {0}.'.format(rtn_code)
        logger.info(msg=log)
        print(log)

    else:
        cmd_err = True
        log = 'Unsuccessfully executed command and returned code: {0}.'.format(rtn_code)
        logger.error(msg=log)
        print(log)

    return cmd_err, rtn_code, std_out
