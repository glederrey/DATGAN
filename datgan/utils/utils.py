#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Other functions
"""


def elapsed_time(delta):
    hours, rem = divmod(delta, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours == 0:
        if minutes == 0:
            return "{:0>2} second{}".format(int(seconds),
                                            's' if int(seconds) > 1 else '')
        else:
            return "{:0>2} minute{} and {:0>2} second{}".format(int(minutes),
                                                                's' if int(minutes) > 1 else '',
                                                                int(seconds),
                                                                's' if int(seconds) > 1 else '')
    else:
        return "{:0>2} hour{}, {:0>2} minute{}, and {:0>2} second{}".format(int(hours),
                                                                            's' if int(hours) > 1 else '',
                                                                            int(minutes),
                                                                            's' if int(minutes) > 1 else '',
                                                                            int(seconds),
                                                                            's' if int(seconds) > 1 else '')

