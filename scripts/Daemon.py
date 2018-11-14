#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


import argparse
import glob
import os
import fcntl
import re
import gc
import socket
import time
import subprocess

import tqdm
import numpy as np
import elektronn2.utils.gpu as gpu


EXP_LOCK = "experiment_start.lock"
EVAL_LOCK = "evaluation_start.lock"

EVAL_EXP_DIR = "~/axon/mkilling/CNN_Training/TCNN/"
EVAL_COMMAND = "python ~/devel/ELEKTRONN2/scripts/TraceEvaluation.py CONFIG --gpu=GPU --modus=mr"

TASK_DIR = "~/axon/mkilling/CNN_Training/tasks/"
REPORT_DEST = "~/axon/mkilling/CNN_Training/all_reports.txt"


# Screen stuff
def get_screenname():
    ret = os.popen('screen -ls').read()
    sessions = re.findall(r"\d+\.pts", ret)
    if len(sessions)>1:
        raise RuntimeError("More than 1 session running")

    return sessions[0]

def decorate_screen_string_window(screenname, id, a):
    return  "screen -x " + screenname + " -p"+ str(id) +" -X stuff " +"'"+   a + "\r'"

def decorate_default_string( screenname, a):
    return "screen -x  " +  screenname  + " -X " + a

def create_window(sname, windowname):
    cmd = [decorate_default_string(sname, " screen -t \"" + windowname + "\" ")]
    subprocess.Popen(cmd, shell=True)
    time.sleep(0.2)

def run_cmd_screen(sname, id, command):
    cmd = [decorate_screen_string_window(sname, id, command)]
    subprocess.Popen(cmd, shell=True)
    time.sleep(0.3)



def create_window_and_run(command, windowname, replacements=None, exec_dir=None):
    sname = get_screenname()


    create_window(sname, windowname)
    run_cmd_screen(sname, windowname, "bash")
    if exec_dir:
        run_cmd_screen(sname, windowname, "cd %s " % exec_dir)

    if replacements:
        for s,t in replacements.items():
            s = str(s)
            t = str(t)
            command = command.replace(s, t)

    run_cmd_screen(sname, windowname, command)


class FSLock(object):
    def __init__(self, dest):
        dest = os.path.expanduser(dest)
        self.f = open(dest, 'a+')
        self.is_acquired = False

    def acquire(self, no_raise=False, print_locker=True):
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.f.write("%s pid=%i\n" % (socket.gethostname(), os.getpid()))
            self.f.flush()
            self.is_acquired = True
            return 0
        except IOError:
            if print_locker:
                try:
                    self.f.seek(0)
                    print("Locked by %s" %self.f.readlines()[-1].strip())
                except IndexError:
                    print("Locked")
            if no_raise:
                return -1
            else:
                raise IOError("Cannot aqcuire lock, is in use")

    def release(self):
        if self.is_acquired:
            self.f.truncate(0)
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.is_acquired = False
        else:
            raise IOError("Cannot relase, was not acquired")

    def __del__(self):
        if self.is_acquired:
            self.release()

        self.f.close()





def search_for_experiments_to_evaluate(base_dir, interval_h=20):
    raise NotImplementedError("Need to fix this for iteration based evaluation")
    base_dir = os.path.expanduser(base_dir)
    experiments = glob.glob1(base_dir, "*")

    todo_list1 = []
    todo_list2 = []

    for exp in experiments:
        eval_dirs = glob.glob1(os.path.join(base_dir, exp), "Eval*")
        eval_hours = []
        for f in eval_dirs:
            try:
                eval_hours.append(re.findall(r"(\d+.\d+)h", f)[0])
            except IndexError:
                pass # regex might not be matching

        eval_hours = np.sort(np.fromiter(eval_hours, dtype=np.float))

        save_files = glob.glob1(os.path.join(base_dir, exp, "Backup"), "*.mdl")
        save_hours = []
        for f in save_files:
            try:
                save_hours.append(re.findall(r"(\d+.\d+)h", f)[0])
            except IndexError:
                pass # regex might not be matching

        #save_hours = [re.findall(r"(\d+.\d+)h", f)[0] for f in save_files]
        save_hours = np.sort(np.fromiter(save_hours, dtype=np.float))

        if len(eval_hours):
            if len(save_hours):
                time_since_last_eval = save_hours[-1] - eval_hours[-1]
                if os.path.exists(
                        os.path.join(base_dir, exp, "0-EVAL-needed")):
                    todo_list1.append(exp)
                elif time_since_last_eval >= interval_h:
                    todo_list2.append(exp)

        else:
            if len(save_hours):
                if os.path.exists(
                        os.path.join(base_dir, exp, "0-EVAL-needed")):
                    todo_list1.append(exp)
                elif (save_hours[-1] >= interval_h):
                    todo_list2.append(exp)

    return todo_list1 + todo_list2


def dict2str(d):
    s = ""
    for k, v in d.items():
        s += "%s %s " % (k, v)

    return s

class TaskList(object):


    def __init__(self):
        self.base_command = None
        self.exec_dir = None
        self.host_num = int(re.findall(r'(\d+)', socket.gethostname())[0])
        self.my_tasks = []
        self.skip_tasks = []
        self.path = None
        self.command = None

    def read(self, path):
        path = os.path.expanduser(path)
        self.path = path
        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                command = re.findall(r'command\s*=\s*(.+)', line)
                exec_dir = re.findall(r'exec_dir\s*=\s*(.+)', line)
                if len(command):
                    self.command = command[0]
                    #print("Command: %s" %command)

                elif len(exec_dir):
                    self.exec_dir = exec_dir[0]
                    #print("Exec_dir: %s" %exec_dir)

                else:
                    line = line.split()
                    task = dict()
                    for i in range(0, len(line), 2):
                        task[line[i]] = line[i + 1]

                    if task['HOST']=='any' or int(task['HOST'])==self.host_num:
                        self.my_tasks.append(task)
                    else:
                        self.skip_tasks.append(task)

        self.my_tasks = self.my_tasks[::-1]


    def start_tasks(self, wait, simulate=False):
        first = True
        while len(self.my_tasks):
            if not first:
                for t in tqdm.trange(wait, 0, -1, leave=False, desc="Waiting"):
                    time.sleep(1)
            else:
                first = False

            free_gpu = gpu.get_free_gpu()
            if free_gpu < 0: # no free gpus, stop
                break

            task = self.my_tasks.pop()
            assert 'GPU' not in task
            task['GPU'] = str(free_gpu)
            task['HOST'] = str(self.host_num)
            print("Starting task %s in %s as %s" %(task, self.exec_dir, self.command))

            if not simulate:
                create_window_and_run(self.command,task["CONFIG"],
                                  task,self.exec_dir)

            task.pop('GPU') # don't write this back
            with open("StartedTasks.txt", 'a') as f:
                if simulate:
                    f.write(dict2str(task) + " - SIMULATED \n")
                else:
                    f.write(dict2str(task) +"\n")

        # Write tasks which are for other hosts and remaining back to file
        if not simulate:
            with open(self.path, 'w') as f:
                f.write("command = %s\n" % self.command)
                f.write("exec_dir = %s\n" % self.exec_dir)
                for task in self.skip_tasks:
                    f.write(dict2str(task) + "\n")
                for task in self.my_tasks:
                    f.write(dict2str(task) + "\n")



def start_experiments(task_file, wait_starts, wait_rerun=120, simulate=False):
    while True:
        lock = FSLock(os.path.join(TASK_DIR, EXP_LOCK))
        lock_state = lock.acquire(no_raise=True)
        if lock_state==0:
            tl = TaskList()
            tl.read(task_file)
            tl.start_tasks(wait_starts, simulate=simulate)
            lock.release()
        else:
            pass
            #print("Skipping because locked")

        gc.collect()

        for t in tqdm.trange(wait_rerun, 0, -1, ncols=0, leave=False, desc="Waiting for Rerun"):
            time.sleep(1)


def start_evaluations(wait_starts, wait_rerun=120, eval_freq=20, simulate=False):
    while True:
        lock = FSLock(os.path.join(TASK_DIR, EVAL_LOCK))
        lock_state = lock.acquire(no_raise=True)
        if lock_state==0:
            todo = search_for_experiments_to_evaluate(EVAL_EXP_DIR, interval_h=eval_freq)
            first = True
            for exp in todo:
                if not first:
                    for t in tqdm.trange(wait_starts, 0, -1, leave=False, desc="Waiting"):
                        time.sleep(1)
                else:
                    first = False

                free_gpu = gpu.get_free_gpu()
                if free_gpu < 0:  # no free gpus, stop
                    break

                task = dict(CONFIG=exp, GPU=free_gpu)
                print("Starting eval %s in %s as %s" %(task, EVAL_EXP_DIR, EVAL_COMMAND))
                if not simulate:
                    create_window_and_run(EVAL_COMMAND, "Eval-"+task["CONFIG"],
                                          task, EVAL_EXP_DIR)

                with open("StartedEvals.txt", 'a') as f:
                    if simulate:
                        f.write(dict2str(task) + " - SIMULATED \n")
                    else:
                        f.write(dict2str(task) + "\n")

            lock.release()

        else:
            pass
            #print("Skipping because locked")

        gc.collect()

        for t in tqdm.trange(wait_rerun, 0, -1, ncols=0, leave=False, desc="Waiting for Rerun"):
            time.sleep(1)

def start_chunky_jobs(script, wait_rerun=30):
    exec_dir, script = os.path.split(os.path.abspath(script))
    while True:
        free_gpu = gpu.get_free_gpu()
        if free_gpu < 0:  # no free gpus, stop
            break
        task = dict(GPU=free_gpu)
        print("Starting Chunk Job on gpu%i"%free_gpu)
        cmd = "python %s run --gpu=GPU"%script
        create_window_and_run(cmd, "ChunkJob",
                            replacements=task, exec_dir=exec_dir)

        for t in tqdm.trange(wait_rerun, 0, -1, ncols=0, leave=False,
                             desc="Waiting for Rerun"):
            time.sleep(1)


# use rather manually and copy result to spreadsheet
def collect_reports(base_dir, report_dest, shadow_processed=True):
    base_dir = os.path.expanduser(base_dir)
    report_dest = os.path.expanduser(report_dest)
    experiments = glob.glob1(base_dir, "*")

    report_strings = []

    for exp in experiments:
        eval_dirs = glob.glob1(os.path.join(base_dir, exp), "Eval*")
        for f in eval_dirs:
            report = glob.glob1(os.path.join(base_dir, exp, f), "*Report*.txt")
            if len(report)==1:
                report_path = os.path.join(base_dir, exp, f, report[0])
                with open(report_path, 'r') as r:
                    lines = r.readlines()
                    if len(lines)!=2:
                        print("%s has not 2 lines in %s" %(exp, report[0]))

                    report_strings.append(lines[-1])

                if shadow_processed:
                    os.rename(report_path, report_path+'~')
                print("FOUND report: %s" % (exp,))
            else:
                if len(report)==0:
                    print("MISSING report: %s" %(exp,))
                else:
                    print("TOO MANY reports: %s" % (exp,))

    with open(report_dest, 'a') as dest:
        dest.writelines(report_strings)

    return report_strings

def parseargs():
   parser = argparse.ArgumentParser(
   usage="Daemon <function> [-s] [--tasks=<taskfile>] [--dest=<reportdest>] [--startwait=<int>] [--daemonwait=<int>]")
   parser.add_argument("function", type=str, choices=['eval', 'expstart', 'collectreports', 'chunky'])
   parser.add_argument("-s", action='store_true') # simulate
   parser.add_argument("--tasks", default="tasks.txt", type=str)  # simulate
   parser.add_argument("--dest", default=REPORT_DEST, type=str)  # destination of report
   parser.add_argument("--startwait", type=int, default=100)
   parser.add_argument("--daemonwait", type=int, default=300)
   parser.add_argument("--evalfreq", type=int, default=20)
   parser.add_argument("--script", type=str, default="")
   parsed = parser.parse_args()
   return parsed.function, parsed.s, parsed.tasks, parsed.dest, parsed.startwait, parsed.daemonwait, parsed.evalfreq, parsed.script


if __name__ == "__main__":
    # if False: # Test
    #     reports = collect_reports(EVAL_EXP_DIR, REPORT_DEST, shadow_processed=True)
    #     start_experiments("tasks.txt", 10, 30, simulate=True)
    #     tdl = search_for_experiments_to_evaluate(EVAL_EXP_DIR)
    #     start_evaluations(10, 30, simulate=True)
   function, s, tasks, dest, startwait, daemonwait, evalfreq, script = parseargs()
   if function=='eval':
       start_evaluations(startwait, daemonwait, simulate=s, eval_freq=evalfreq)
   elif function=='expstart':
       start_experiments(tasks, startwait, daemonwait, simulate=s)
   elif function=='collectreports':
       shadow_processed = not s
       collect_reports(EVAL_EXP_DIR, dest, shadow_processed=shadow_processed)
   elif function=='chunky':
       assert script != ""
       start_chunky_jobs(script, wait_rerun=daemonwait)

