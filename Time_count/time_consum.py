# ------------------------------------------------------------------
# FileName: time_consum
# Author: cong
# CreateTime: 2021/12/20 下午 9:08
# Description:
# ------------------------------------------------------------------
import datetime
import time


def get_Time_difference(startTime, endTime):
    '''计算两个时间点之间的分钟数'''
    # 处理格式,加上秒位
    startTime1 = startTime
    endTime1 = endTime
    # 计算分钟数
    startTime2 = datetime.datetime.strptime(startTime1, "%Y-%m-%d %H:%M:%S")
    endTime2 = datetime.datetime.strptime(endTime1, "%Y-%m-%d %H:%M:%S")
    seconds = (endTime2 - startTime2).seconds
    # 来获取时间差中的秒数。注意，seconds获得的秒只是时间差中的小时、分钟和秒部分的和，并没有包含时间差的天数（既是两个时间点不是同一天，失效）
    total_seconds = (endTime2 - startTime2).total_seconds()
    # 来获取准确的时间差，并将时间差转换为秒
    # print(total_seconds)
    mins = int(total_seconds / 60)
    hours = int(mins / 60)
    return hours, mins, int(total_seconds)


def show_Time_consuming(startTime, stop_tn):
    fmt1 = "%Y-%m-%d %H:%M:%S"
    Date = time.strftime(fmt1, time.localtime(time.time()))  # 把传入的元组按照格式，输出字符串
    startTime_1 = startTime
    endTime_1 = Date
    hours, mins, seconds = get_Time_difference(startTime_1, endTime_1)
    time_consum_str = str(hours) + 'h-' + str(mins - hours * 60) + 'm-' + str(seconds - mins * 60) + 's'
    if hours >= stop_tn:
        stop_f = 1
    else:
        stop_f = 0
    return time_consum_str, stop_f


if __name__ == "__main__":
    Date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))  # 把传入的元组按照格式，输出字符串

    # startTime = '2020-11-30 16:30:00'
    startTime = Date
    print(show_Time_consuming(startTime, 10))
