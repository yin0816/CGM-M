# ------------------------------------------------------------------
# FileName: log_save
# Author: cong
# CreateTime: 2021/7/6 上午 10:14
# Description:
# ------------------------------------------------------------------
import sys
import os


class Logger(object):

    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# sys.stdout = Logger('logs.txt', sys.stdout)  # 控制台输出日志
# sys.stderr = Logger('logs.txt', sys.stderr)  # 错误输出日志

path = os.path.abspath(os.path.dirname(__file__))
# type = sys.getfilesystemencoding()
# sys.stdout = Logger('./logs.txt')

# print(path)
# print(os.path.dirname(__file__))
# print('------------------')

