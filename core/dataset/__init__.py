import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


filedir = os.path.dirname(__file__)

for p in ["utils", "."]:
    add_path(
        os.path.join(filedir, os.pardir, p)
    )
