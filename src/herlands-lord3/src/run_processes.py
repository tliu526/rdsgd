'''
General function for taking in a text file with lines to run and running them. Developed based on:
http://stackoverflow.com/questions/18123325/always-run-a-constant-number-of-subprocesses-in-parallel

Sample usage:

    python run_processes.py list_of_commands.txt 5
'''

import subprocess
import time
import sys

NextPSNo = 0
MaxConcurrentPS = 1
MaxPS = 100000  # Note this would be better to be len(pslist)
Processes = []

def StartNew(pslist):
    """ Start a new subprocess if there is work to do """
    global NextPSNo
    global Processes

    if NextPSNo < MaxPS:
        #proc = subprocess.Popen(['python', 'script.py', pslist[NextPSNo], OnExit])
        print ("Starting Process %s...", pslist[NextPSNo])
        proc = subprocess.Popen(pslist[NextPSNo].split(' '))
        print ("Started Process %s", pslist[NextPSNo])
        NextPSNo += 1
        Processes.append(proc)


def CheckRunning(pslist):
    """ Check any running processes and start new ones if there are spare slots."""
    global Processes
    global NextPSNo

    for p in reversed(range(len(Processes))):  # Check the processes in reverse order
        if Processes[p].poll() is not None: # If the process hasn't finished will return None
            del Processes[p] # Remove from list - this is why we needed reverse order

    while (len(Processes) < MaxConcurrentPS) and (NextPSNo < MaxPS): # More to do and some spare slots
        StartNew(pslist)
        time.sleep(5)



def get_process_list(sys):
    global MaxConcurrentPS

    if len(sys.argv) < 2:
        assert False, 'Error: Must provide at least 1 argument which is the filename'
    elif len(sys.argv) > 3:
        assert False, 'Error: Max 2 arguments'

    # Open file
    with open(sys.argv[1]) as f:
        f = f.readlines()

    # Get max processes
    if len(sys.argv) == 3:
        MaxConcurrentPS = int(sys.argv[2])

    # remove whitespace characters like `\n` at the end of each line
    #pslist = [x.strip() for x in f]
    pslist = []
    for line in f:
        line = line.strip()
        if line != '': pslist.append(line)

    return pslist



if __name__ == "__main__":
    global MaxPS

    pslist = get_process_list(sys)
    MaxPS = len(pslist)

    CheckRunning(pslist) # This will start the max processes running
    while (len(Processes) > 0): # Some thing still going on.
        time.sleep(60.0) # You may wish to change the time for this
        print 'Processes:', Processes
        CheckRunning(pslist)

    print ("Done!")
