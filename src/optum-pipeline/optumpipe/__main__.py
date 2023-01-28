"""
"Main" script file for optum-pipeline. 

Initializes instances of OptumPipe and runs them via argparse.
"""

import argparse

from .pipe import *

def main():
    """"""
    parser = argparse.ArgumentParser(description="Schedules data pulls and merges for Optum 2020 data")

    parser.add_argument("--task", type=str, choices=SUPPORTED_TASKS,
                        help="Specify which RDD task to run")
    parser.add_argument("--table", type=str, choices=SUPPORTED_TABLE_TYPES,
                        help="Specify which table type to target")
    parser.add_argument("--test", action="store_true",
                        help="optionally schedule a test run")
    args = parser.parse_args()

    pipe = OptumPipe(args.task, args.table, test=args.test)
    pipe.run()

if __name__ == "__main__":
    main()
