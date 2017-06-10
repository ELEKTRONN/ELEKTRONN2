# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import code
import getpass
import logging
import os
import shutil
import socket
import traceback
import datetime
from itertools import repeat
from multiprocessing import Pool
from os.path import abspath, dirname

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz as integrate
import theano
import theano.sandbox.cuda

from .. import utils
from ..config import config, change_logging_file
from ..data import image
from ..utils import plotting
from ..utils.locking import FileLock


logger = logging.getLogger('elektronn2log')
inspection_logger = logging.getLogger('elektronn2log-inspection')
user_name = getpass.getuser()
# Setup for prompt_toolkit interactive shell
shortcut_completions = [  # Extra words to register completions for:
    'q', 'kill', 'sethist', 'setlr', 'setmom', 'setwd', 'sf', 'preview',
    'paramstats', 'gradstats', 'actstats', 'debugbatch', 'load']
import prompt_toolkit
from  prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from ..utils.ptk_completions import NumaCompleter


ptk_hist = InMemoryHistory()


def user_input(local_vars):
    save_name = local_vars['exp_config'].save_name
    _banner = """
    ========================
    === ELEKTRONN2 SHELL ===
    ========================
    >> %s <<
    Shortcuts:
    'help' (display this help text),
    'q' (leave menu),         'kill'(saving last params),
    'sethist <int>',          'setlr <float>',
    'setmom <float>',         'setwd <float> (weight decay)
    'paramstats' ,            'gradstats',
    'actstats' (print statistics)
    'sf <nodename>' (show filters)',
    'load <filename>' (param files only, no model files),
    'preview' (produce preview predictions),
    'ip' (start embedded IPython shell)

    For everything else enter a command in the command line\n""" % (save_name,)

    _ipython_banner = """    You are now in the embedded IPython shell.
    You still have full access to the local scope of the ELEKTRONN2 shell
    (e.g. 'model', 'batch'), but shortcuts like 'q' no longer work.

    To leave the IPython shell and switch back to the ELEKTRONN2 shell, run
    'exit()' or hit 'Ctrl-D'."""

    print(_banner)
    data = local_vars['data']
    batch = local_vars['batch']
    trainer = local_vars['self']
    model = trainer.model
    exp_config = local_vars['exp_config']
    local_vars.update(locals())  # put the above into scope of console
    console = code.InteractiveConsole(locals=local_vars)

    while True:
        try:
            try:
                inp = prompt_toolkit.prompt(u"%s@neuromancer: " % user_name,
                                            # needs to be an explicit ustring for py2-compat
                                            history=ptk_hist,
                                            completer=NumaCompleter(
                                                lambda: local_vars, lambda: {},
                                                words=shortcut_completions,
                                                words_metastring='(shortcut)'),
                                            auto_suggest=AutoSuggestFromHistory())
            # Catch all exceptions in order to prevent catastrophes in case ptk suddenly breaks
            except Exception:
                inp = console.raw_input("%s@neuromancer: " % user_name)
            logger.debug('(Shell received command "{}")'.format(inp))
            if inp=='q':
                break
            elif inp=='kill':
                break
            elif inp == 'help':
                print(_banner)
            elif inp == 'ip':
                try:
                    import IPython
                    IPython.embed(header=_ipython_banner)
                except ImportError:
                    print('IPython is not available. You will need to install '
                          'it to use this function.')
            elif inp.startswith('sethist'):
                i = int(inp.split()[1])
                exp_config.history_freq = i
            elif inp.startswith('setlr'):
                i = float(inp.split()[1])
                model.lr = i
            elif inp.startswith('setmom'):
                i = float(inp.split()[1])
                model.mom = i
            elif inp.startswith('setwd'):
                i = float(inp.split()[1])
                model.wd = i
            elif inp.startswith('sf'):
                try:
                    name = inp.split()[1]
                    w = model[name].w.get_value()
                    m = plotting.embedfilters(w)
                    with FileLock('plotting'):
                        plt.imsave('filters_%s.png' % name, m, cmap='gray')
                except:  # try to print filter of first Layer with w
                    for name, node in model.nodes.items():
                        if hasattr(node, 'w'):
                            m = plotting.embedfilters(node.w.get_value())
                            with FileLock('plotting'):
                                plt.imsave('filters.png', m, cmap='gray')
                            break

            elif inp=='preview':
                try:
                    trainer.preview_slice(**exp_config.preview_kwargs)
                except Exception:
                    traceback.print_exc()
                    print('\n\nPlease check if/how you have configured previews '
                          'in your config file.\n(Look for "preview_data_path" '
                          'and "preview_kwargs" variables.)')
            elif inp=='paramstats':
                model.paramstats()
            elif inp=='gradstats':
                model.gradstats(*batch)
            elif inp=='actstats':
                model.actstats(*batch)
            elif inp=='debugbatch':
                trainer.debug_getcnnbatch()
            elif inp.startswith('load'):
                file_path = inp.split()[1]
                params = utils.pickleload(file_path)
                model.set_param_values(params)
            else:
                console.push(inp)

            plt.pause(0.00001)
        except KeyboardInterrupt:
            print('Enter "q" to leave the shell and continue training.\n'
                  'Enter "kill" to kill the training, saving current parameters.')
        except IndexError as err:
            if any([inp.startswith(shortcut) for shortcut in shortcut_completions]):  # ignore trailing spaces
                print('IndexError. Probably you forgot to type a value after the shortcut "{}".'.format(inp))
            else:
                raise err  # All other IndexErrors are already correctly handled by the console.
        except ValueError as err:
            if any([inp.startswith(shortcut) for shortcut in shortcut_completions]):  # ignore trailing spaces
                print('ValueError. The "{}" shortcut received an unexpected argument.'.format(inp))
            else:
                raise err  # All other IndexErrors are already correctly handled by the console.
        except Exception:
            traceback.print_exc()
            print('\n\nUnhandled exception occured. See above traceback for debug info.\n'
                  'If you think this is a bug, please consider reporting it at '
                  'https://github.com/ELEKTRONN/ELEKTRONN2/issues.')

    return inp


class Schedule(object):
    """
    Create a schedule for parameter or property

    Examples
    --------

    >>> lr_schedule = Schedule(dec=0.95) # decay by factor 0.95 every 1000 steps (i.e. decreasing by 5%)
    >>> wd_schedule = Schedule(lindec=[4000, 0.001]) # from 0.001 to 0 in 400 steps
    >>> mom_schedule = Schedule(updates=[(500,0.8), (1000,0.7), (1500,0.9), (2000, 0.2)])
    >>> dropout_schedule = Schedule(updates=[(1000,[0.2, 0.2])]) # set rates per Layer

    """  # TODO Update examples with actual usage of ``schedules`` dict in configs

    def __init__(self, **kwargs):
        ###TODO setting of categorical values (True/False) via 'updates'
        # multiplicative decay
        self._target = None
        self.variable_getter = None
        self.variable_setter = None
        if 'dec' in kwargs:
            self.mode = 'mult'
            self.factor = float(kwargs['dec'])
            self.next_update = 1000
        # updates to certain values at certain times
        elif 'updates' in kwargs:
            self.factor = 1.0
            self.mode = 'set'
            self.update_steps = [up[0] for up in kwargs['updates']]
            self.update_values = [up[1] for up in kwargs['updates']]
            self.next_update = min(1000, self.update_steps[0])
        # linear decay to 0 after n steps
        elif 'lindec' in kwargs:
            self.factor = 1.0
            self.mode = 'lin'
            init_val = kwargs['lindec'][1]
            self.n_steps = kwargs['lindec'][0]  # number of steps until 0
            self.delta = (init_val * 1000.0 / self.n_steps)
            self.next_update = 1000
        else:
            raise ValueError("Unknown schedule args %s" % (kwargs))

    def update(self, iteration):
        if self.mode=='mult':
            self.variable_setter(
                np.multiply(self.variable_getter(), self.factor))
            self.next_update = iteration + 1000

        elif self.mode=='set':
            if iteration==self.update_steps[
                0]:  # process value from schedule list
                self.variable_setter(self.update_values[0])
                self.update_steps.pop(0)
                self.update_values.pop(0)
                if len(self.update_steps)==0:  # schedule finished
                    if np.allclose(self.factor, 1.0):  # if factor 1 do nothing
                        self.next_update = -1
                    else:  # From now on only mult decay
                        self.next_update = iteration + 1000
                        self.mode = 'mult'
                elif np.allclose(self.factor, 1.0):
                    self.next_update = self.update_steps[0]
                else:
                    self.next_update = min(iteration + 1000,
                                           self.update_steps[0])
            else:  # make multiplicative update
                self.variable_setter(
                    np.multiply(self.variable_getter(), self.factor))
                if len(self.update_steps)==0:  # only mult decay
                    self.next_update = iteration + 1000
                elif np.allclose(self.factor, 1.0):
                    self.next_update = self.update_steps[0]
                else:
                    self.next_update = min(iteration + 1000,
                                           self.update_steps[0])

        elif self.mode=='lin':
            if iteration < self.n_steps:
                self.variable_setter(
                    np.subtract(self.variable_getter(), self.delta))
                self.next_update = iteration + 1000
            elif iteration==self.n_steps:
                self.variable_setter(0.0)
                self.next_update = -1

    def bind_variable(self, variable_param=None, obj=None, prop_name=None):
        # variable_param is like theano.SharedVariable
        if variable_param is not None and hasattr(variable_param,
                                                  'get_value') and hasattr(
            variable_param, 'set_value'):
            self.variable_getter = variable_param.get_value
            self.variable_setter = variable_param.set_value
            self._target = variable_param.name

        # This is (most likely a theano.SharedVariable) wrapped as property of model-obj
        elif obj is not None and prop_name is not None:
            if hasattr(obj.__class__,
                       prop_name):  # it is a property, get the s/getters
                prop = getattr(obj.__class__, prop_name)
                self.variable_getter = lambda: prop.fget(obj)
                self.variable_setter = lambda val: prop.fset(obj, val)
                self._target = prop_name
            if hasattr(obj,
                       prop_name):  # it is a attribute, create the s/getters
                prop = getattr(obj, prop_name)
                self.variable_getter = lambda: getattr(obj, prop_name)
                self.variable_setter = lambda val: setattr(obj, prop_name, val)
                self._target = prop_name
            else:  # search in nontrainable params
                if hasattr(obj, 'nontrainable_params'):
                    # print("%s is not a property, searching in non-trainable params!" %prop_name)
                    try:
                        variable_param = obj.nontrainable_params[prop_name]
                        self.variable_getter = variable_param.get_value
                        self.variable_setter = variable_param.set_value
                        self._target = variable_param.name
                        logger.debug(
                            "Found %s in model.nontrainable_params" % variable_param.name)
                    except:
                        raise AttributeError(
                            "%s not found or is not VariableParam!" % prop_name)
                else:
                    raise AttributeError(
                        "%s is not a property/attribute of %s "
                        "and not in model.nontrainable_params!" % (
                            prop_name, obj.__class__))

        else:
            raise ValueError()

    def __repr__(self):
        s = "Schedule for %s:" \
            "  mode=%s, decay-factor=%f, next update=%i\n" % (
                self._target, self.mode, self.factor, self.next_update)
        if hasattr(self, 'delta'):
            s += "  delta=%f\n" % (self.delta,)
        if hasattr(self, 'update_steps'):
            s += "  update_steps=%s, update_vals=%s\n" % (
                self.update_steps, self.update_values)
        return s


def loadhistorytracker(file_name):
    ht = HistoryTracker()
    ht.load(file_name)
    return ht


class HistoryTracker(object):
    def __init__(self):
        self.plotting_proc = None
        self.debug_outputs = None
        self.regression_track = None
        self.debug_output_names = None

        self.timeline = utils.AccumulationArray(n_init=int(1e5), dtype=dict(
            names=[u'time', u'loss', u'batch_char', ], formats=[u'f4', ] * 3))

        self.history = utils.AccumulationArray(n_init=int(1e4), dtype=dict(
            names=[u'steps', u'time', u'train_loss', u'valid_loss',
                   u'loss_gain', u'train_err', u'valid_err', u'lr', u'mom',
                   u'gradnetrate'], formats=[u'i4', ] + [u'f4', ] * 9))

    def update_timeline(self, vals):
        self.timeline.append(vals)

    def register_debug_output_names(self, names):
        self.debug_output_names = names

    def update_history(self, vals):
        self.history.append(vals)

    def update_debug_outputs(self, vals):
        if self.debug_outputs is None:
            self.debug_outputs = utils.AccumulationArray(n_init=int(1e5),
                                                         right_shape=len(vals))

        self.debug_outputs.append(vals)

    def update_regression(self, pred, target):
        if self.regression_track is None:
            assert len(pred)==len(target)
            p = utils.AccumulationArray(n_init=int(1e5), right_shape=len(pred))
            t = utils.AccumulationArray(n_init=int(1e5), right_shape=len(pred))
            self.regression_track = [p, t]

        self.regression_track[0].append(pred)
        self.regression_track[1].append(target)

    def save(self, save_name):
        file_name = save_name + '.history.pkl'
        utils.picklesave([self.timeline, self.history, self.debug_outputs,
                          self.debug_output_names, self.regression_track],
                         file_name)

    def load(self, file_name):
        (self.timeline, self.history, self.debug_outputs,
         self.debug_output_names, self.regression_track) = utils.pickleload(
            file_name)

    def plot(self, save_name=None, autoscale=True, close=True):
        #        if self.plotting_proc is not None:
        #            self.plotting_proc.join()
        #
        #        self.plotting_proc = Process(target=self._plot,
        #                                     args=(save_name, autoscale))
        #        self.plotting_proc.start()

        save_name = "0-" + save_name
        plotting.plot_hist(self.timeline, self.history, save_name,
                           config.loss_smoothing_length, autoscale)

        if self.debug_output_names and self.debug_outputs.length:
            plotting.plot_debug(self.debug_outputs, self.debug_output_names,
                                save_name)

        if self.regression_track:
            plotting.plot_regression(self.regression_track[0],
                                     self.regression_track[1], save_name)
            plotting.plot_kde(self.regression_track[0],
                              self.regression_track[1], save_name)

        if close:
            plt.close('all')


class ExperimentConfig(object):
    @classmethod
    def levenshtein(cls, s1, s2):
        """
        Computes Levenshtein-distance between ``s1`` and ``s2`` strings
        Taken from: http://en.wikibooks.org/wiki/Algorithm_Implementation/
        Strings/Levenshtein_distance#Python
        """
        if len(s1) < len(s2):
            return cls.levenshtein(s2, s1)
        # len(s1) >= len(s2)
        if len(s2)==0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row
                # are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1  # than s2
                substitutions = previous_row[j] + (c1!=c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def __init__(self, exp_file, host_script_file=None,
                 use_existing_dir=False):
        self.exp_file = os.path.expanduser(exp_file)
        self.host_script_file = host_script_file
        self.use_existing_dir = use_existing_dir

        # Field to be provided by user
        self.save_name = None
        self.save_path = None
        self.create_model = None
        self.model_load_path = None  # alternative to above
        self.model_load_args = None  # to override mfp, inputsize etc
        self.batch_size = None  # try to infer from model itself
        self.preview_data_path = None
        self.preview_kwargs = None
        self.data_class = None  # <String>: Name of Data Class in TrainData or
        # <tuple>: (path_to_file, class_name)
        self.data_init_kwargs = dict()
        self.data_batch_args = dict()  # NOT batch_size and NOT source!

        self.n_steps = None
        self.max_runtime = None
        self.history_freq = None
        self.monitor_batch_size = None

        self.optimiser = None
        self.optimiser_params = None

        self.lr_schedule = None
        self.wd_schedule = None
        self.mom_schedule = None
        self.dropout_schedule = None
        self.gradnet_schedule = None
        self.schedules = dict()

        self.class_weights = None
        self.lazy_labels = None

        self.sequence_training = None

        self.read_user_config()
        if not use_existing_dir:
            self.make_dir()

        # Set log file to save path:
        old_lfile_handler = \
            [x for x in logger.handlers if isinstance(x, logging.FileHandler)][
                0]

        logger.removeHandler(old_lfile_handler)
        # TODO: Transfer log entries from old log file to new one?

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, mode=0o755)

        lfile_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s]\t%(message)s', datefmt='%H:%M')
        lfile_path = os.path.join(self.save_path,
                                  '%s.log' % ('0-' + self.save_name,))
        lfile_level = logging.DEBUG
        lfile_handler = logging.FileHandler(lfile_path)
        lfile_handler.setLevel(lfile_level)
        lfile_handler.setFormatter(lfile_formatter)
        logger.addHandler(lfile_handler)

        # And for inspection log too
        change_logging_file(inspection_logger, self.save_path,
                            file_name='%s.inspection-log' % (
                                '0-' + self.save_name,))

        host_name = socket.gethostname()
        now = datetime.datetime.today().isoformat()
        device = 'CPU'
        try:
            if theano.sandbox.cuda.cuda_enabled:
                device = 'GPU {}'.format(
                    theano.sandbox.cuda.active_device_number())
        except:
            pass
        logger.info('Running on {}@{}, using {}. Start time: {}'.format(
            user_name, host_name, device, now))

    def read_user_config(self):
        logger.info("Reading exp_config-file %s" % (self.exp_file,))
        allowed = list(self.__dict__.keys()) + ['np', ]
        custom_dict = dict()
        exec (compile(open(self.exp_file).read(), self.exp_file, 'exec'), {},
              custom_dict)
        all_names = allowed + list(config.__dict__.keys())
        strange_keys = []  # list of keys that are not allowed for the config.
        didyoumean = []  # list of tuples that contain strange keys and their respective suggested spelling.
        for key in custom_dict:
            if key in allowed:
                if key in ['save_path', 'model_load_path', 'preview_data_path',
                           'data_class']:
                    try:
                        if key=='data_class' and isinstance(custom_dict[key],
                                                            tuple):
                            custom_dict[key][0] = os.path.expanduser(
                                custom_dict[key][0])
                        else:
                            custom_dict[key] = os.path.expanduser(
                                custom_dict[key])
                    except:
                        pass

                setattr(self, key, custom_dict[key])
            elif hasattr(config, key):
                setattr(config, key, custom_dict[key])
                if getattr(config, key)!=custom_dict[key]:
                    logger.info(
                        'Overriding default config "%s" (%s) with value: %s' % (
                            key, getattr(config, key), custom_dict[key]))

            else:
                strange_keys.append(str(key))

        for key in strange_keys:
            min_dist = np.argmin([self.levenshtein(key, x) for x in all_names])
            didyoumean.append((key, all_names[min_dist]))

        if didyoumean:
            logger.warning('Unknown config variables have been found:')
            for key, suggestion in didyoumean:
                logger.warning(
                    '"{}":\tbest match: "{}"?'.format(key, suggestion))

        if self.save_path:
            path = self.save_path
        else:
            path = config.save_path

        # If save_name is None, use the config file name without its extension, (e.g. '/cnnconf/mycnn.py' -> 'mycnn')
        if self.save_name is None:
            fname = os.path.basename(self.exp_file)
            fname_without_ext = os.path.splitext(fname)[0]
            timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
            self.save_name = fname_without_ext + '__' + timestamp
            logger.info('Auto-assigned save_name = "{}"'.format(self.save_name))

        self.save_path = os.path.join(path, self.save_name)
        logger.info('Writing files to directory "{}"'.format(self.save_path))

        self.check_config()

        self.create_model.__globals__.update(custom_dict)  # this is necessary
        # to make variable in the config file usable in create_model (as it
        # would naturally be expected by humans)

    def check_config(self):
        for k, v in self.__dict__.items():
            if k=='create_model' and v is None:
                if self.__dict__['model_load_path'] is None:
                    raise ValueError("Invalid experiment config: one of "
                                     "'create_model' or 'model_load_path' "
                                     "must be given.")
            elif k=='model_load_path' and v is None:
                if self.__dict__['create_model'] is None:
                    raise ValueError("Invalid experiment config: one of "
                                     "'create_model'or 'model_load_path' "
                                     "must be given.")
            elif v is None:  # list of variables for which none is allowed
                if k not in ['class_weights', 'lazy_labels',
                             'gradnet_schedule', 'dropout_schedule',
                             'lr_schedule', 'wd_schedule', 'mom_schedule',
                             'schedules', 'preview_kwargs',
                             'preview_data_path', 'model_load_args',
                             'sequence_training']:
                    raise ValueError("'%s' must not be 'None'" % (k,))

    def make_dir(self):
        """
        Saves all python files into the folder specified by ``self.save_path``
        Also changes working directory to the ``save_path`` directory
        """
        if os.path.exists(self.save_path):
            if config.overwrite:
                logger.info("Overwriting existing save directory: %s" % (
                    self.save_path,))
                shutil.rmtree(self.save_path)
            else:
                raise RuntimeError('The save directory does already exist!')

        os.makedirs(self.save_path, mode=0o755)
        os.mkdir(os.path.join(self.save_path, 'Backup'), 0o755)

        # Backup config
        name = os.path.split(self.exp_file)[1]  # e.g. Experiment1_conf.py
        shutil.copy(self.exp_file, os.path.join(self.save_path, '0-' + name))
        os.chmod(os.path.join(self.save_path, '0-' + name), 0o755)

        if self.host_script_file:
            name = os.path.split(self.host_script_file)[1]
            shutil.copy(self.host_script_file,
                        os.path.join(self.save_path, "Backup", name))
            os.chmod(os.path.join(self.save_path, "Backup", name), 0o755)

        if config.backupsrc:
            # Save the full source code that was used to
            # train the network (takes about 180ms).
            try:
                # Assuming that the __file__ is 2 levels below the package root
                pkgpath = dirname(dirname(abspath(__file__)))
                backuppath = os.path.join(self.save_path, 'Backup', 'src')
                shutil.make_archive(backuppath, 'gztar', pkgpath)
                logger.info(
                    'Archived source code in {}.tar.gz'.format(backuppath))
            except Exception:
                logger.warning('Failed to archive the package source code '
                               'in {}/Backup/'.format(self.save_path))


### Testing / Evaluation ######################################################
def confusion_table(labs, preds):
    """
    Gives all counts of binary classifications situations:
      :labs: correct labels (-1 for ignore)
      :preds: 0 for negative 1 for positive (class probabilities must be thresholded first)
    Return:
      :count of: (true positive, true negative, false positive, false negative)
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    for lab, pred in zip(labs, preds):
        tp += np.count_nonzero((lab * pred)[lab >= 0])
        tn += np.count_nonzero(((1 - lab) * (1 - pred))[lab >= 0])
        fp += np.count_nonzero(((1 - lab) * pred)[lab >= 0])
        fn += np.count_nonzero((lab * (1 - pred))[lab >= 0])
    return tp, tn, fp, fn


def performance_measure(tp, tn, fp, fn):
    """
    For output of confusion table gives various perfomance performance_measures:
      :return: tpr, fpr, precision, recall, balanced accuracy, accuracy, f1-score
    """
    tpr = float(tp) / (tp + fn) if (tp + fn)!=0 else 0  # true positive rate
    fpr = float(fp) / (fp + tn) if (fp + tn)!=0 else 0  # false positive rate
    tnr = float(tn) / (tn + fp) if (tn + fp)!=0 else 0
    recall = float(tp) / (tp + fn) if (tp + fn)!=0 else 0
    precision = float(tp) / (tp + fp) if (tp + fp)!=0 else 1
    bal_accur = 0.5 * tpr + 0.5 * tnr  # balanced accuracy
    accur = float(tp + tn) / (tp + tn + fp + fn)
    if (precision + recall)!=0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return tpr, fpr, precision, recall, bal_accur, accur, f1


def roc_area(tpr, fpr):
    """
    Integrate ROC curve:
      :data: (tpr, fpr)
      :return: area
    """
    return integrate(tpr[::-1], fpr[::-1])[-1]


def eval_thresh(args):
    """
    Calculates various performance measures at certain threshold
    :param args: thresh, labs, preds
    :return: tpr, fpr, precision, recall, bal_accur, accur, f1
    """
    thresh, labs, preds = args
    classification = (preds >= thresh).astype('int')
    tp, tn, fp, fn = confusion_table(labs, classification)
    # tpr, fpr, precision, recall, bal_accur, accur, f1
    perf = performance_measure(tp, tn, fp, fn)
    print("thresh=%.2f, acc=%.4f, prec=%.4f, recall=%.4f, F1=%.4f" % (
        thresh, perf[5], perf[2], perf[3], perf[6]))
    return perf


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def evaluate(gt, preds, save_name, thresh=None, n_proc=None):
    """
    Evaluate prediction w.r.t. GT
    Saves plot to file
    :param save_name:
    :param gt:
    :param preds: from 0.0 to 1.0
    :param thresh: if thresh is given (e.g. from tuning on validation set)
    some performance measures are shown at this threshold
    :return: perf, roc-area, threshs
    """
    n = 64
    threshs = np.linspace(0, 1, n)
    perf = np.zeros((7, threshs.size))
    print("Scanning for best probmap THRESHOLD")
    if n_proc:
        if n_proc > 2:
            mp = Pool(6)
            ret = mp.imap(eval_thresh, zip(threshs, repeat(gt), repeat(preds)))
    else:
        ret = list(map(eval_thresh, zip(threshs, repeat(gt), repeat(preds))))

    for i, r in enumerate(ret):
        perf[:, i] = r

    # Find thresh according to maximal accuracy
    thresh = find_nearest(threshs, thresh) if thresh else threshs[
        perf[5, :].argmax()]

    area = roc_area(perf[0, :], perf[1, :])
    area2 = roc_area(perf[2, :], perf[3, :])

    plt.figure(figsize=(12, 9))

    plt.subplot(221)
    plt.plot(threshs, perf[6, :].T)
    plt.ylim(0, 1)
    f1_max = perf[6, np.where(threshs==thresh)]
    plt.vlines(thresh, 0, 1, color='gray')
    plt.title("F1=%.2f at %.4f" % (f1_max, thresh))
    plt.xlabel("Classifier Threshold")

    plt.subplot(222)
    plt.plot(threshs, perf[5, :].T)
    plt.ylim(0, 1)
    acc_max = perf[5, np.where(threshs==thresh)]
    plt.vlines(thresh, 0, 1, color='gray')
    plt.title("Accuracy max=%.2f at %.4f" % (acc_max, thresh))
    plt.xlabel("Classifier Threshold")

    plt.subplot(223)
    plt.plot(perf[3, :].T, perf[2, :].T)
    plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall AUC=%.4f" % (area2,))

    plt.subplot(224)
    plt.plot(perf[1, :].T, perf[0, :].T)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = "ROC AUC=%.4f" % (area,)
    plt.title(title)

    with FileLock('plotting'):
        plt.savefig(save_name + ".performance.png", bbox_inches='tight')

    return acc_max, area, thresh


def error_hist(gt, preds, save_name, thresh=0.42):
    """
    preds: predicted probability of class '1'
    Saves plot to file
    """
    n_bins = min(500, preds.size ** 1.0 / 3)
    tot_hist, bins = np.histogram(preds.ravel(), bins=n_bins, range=(0, 1))
    bin_diff = bins[1] - bins[2]
    bins = bins[:-1] + 0.5 * bin_diff

    pos_hist, _ = np.histogram(preds[gt==1], bins=n_bins, range=(0, 1))
    neg_hist, _ = np.histogram(preds[gt==0], bins=n_bins, range=(0, 1))

    plt.figure(figsize=(12, 9))
    plt.plot(bins, tot_hist, 'k', label='total')
    plt.plot(bins, pos_hist, 'g', label='pos GT label')
    plt.plot(bins, neg_hist, 'r', label='neg GT label')
    plt.vlines(thresh, 0, np.max(pos_hist), color='gray')

    plt.legend(loc=9)
    plt.semilogy()
    plt.xlim(0, 1)
    plt.title('Error Histogram')
    with FileLock('plotting'):
        plt.savefig(save_name + ".error_histogram.png", bbox_inches='tight')


def rescale_fudge(pred, fudge=0.15):
    pred[pred < fudge] = fudge
    pred[pred > (1 - fudge)] = (1.0 - fudge)
    pred -= fudge
    pred *= 1.0 / (1 - 2 * fudge)
    return pred


def binary_nll(pred, gt):
    if isinstance(pred, (list, tuple)):
        nlls = [binary_nll(x[0], x[1]) for x in zip(pred, gt)]
        return np.mean(nlls)

    nll_neg = (1 - gt) * np.log(1 - pred)
    nll_neg[np.isclose(gt, 0)] = 0

    nll_pos = gt * np.log(pred)
    nll_pos[np.isclose(gt, 1)] = 0

    nll = -0.5 * (nll_pos.mean() + nll_neg.mean())
    return nll


def evaluate_model_binary(model, name, data=None, valid_d=None, valid_l=None,
                          train_d=None, train_l=None, n_proc=2, betaloss=False,
                          fudgeysoft=False):
    if not model.prediction_node.shape['f']==2:
        logger.warning("Evaluate_model_binary is intended only for binary"
                       "classification, this model has more or less outputs than 2")

    report_str = "T_nll,\tT_acc,\tT_ROCA,\tV_nll,\tV_acc,\tV_ROCA,\td_acc,\t" \
                 "d_ROCA,\tri0,\tr01,\tri2,\tri3,\trim\n"

    # Training Data ###########################################################
    if train_d is None:
        train_d = data.train_d
    if train_l is None:
        train_l = data.train_l
    train_preds = []
    train_gt = []
    for i, (d, l) in enumerate(zip(train_d[:4], train_l[:4])):
        if os.path.exists("2-" + name + "_train_%i_pred.h5" % i):
            pred = utils.h5load("2-" + name + "_train_%i_pred.h5" % i)
        else:
            pred = model.predict_dense(d, pad_raw=False)  # (f,z,x,y)
            utils.h5save(pred, "2-" + name + "_train_%i_pred.h5" % i)

        if betaloss:
            pred = pred[0]  # only mode
        else:
            pred = pred[1]  # only pred for class '1'

        l = l[0]  # throw away channel
        l, pred = image.center_cubes(l, pred, crop=True)
        train_preds.append(pred)
        train_gt.append(l)

    train_gt = [gt > 0.5 for gt in
                train_gt]  # binarise possibly probabilistic GT

    train_acc, train_area, train_thresh = evaluate(train_gt, train_preds,
                                                   "1-" + name + "_train")

    gt_flat = np.concatenate(list(map(np.ravel, train_gt)))
    preds_flat = np.concatenate(list(map(np.ravel, train_preds)))
    if fudgeysoft:
        train_nll = binary_nll(rescale_fudge(preds_flat), gt_flat)
    else:
        train_nll = binary_nll(preds_flat, gt_flat)

    print("Train nll %.3f" % train_nll)
    report_str += "%.3f,\t%.3f,\t%.3f,\t" % (train_nll, train_acc, train_area)

    error_hist(gt_flat, preds_flat, "1-" + name + "_train",
               thresh=train_thresh)

    # Validation data #########################################################
    if data and len(data.valid_l)==0:
        raise RuntimeError("No validation data!")

    if valid_d is None:
        valid_d = data.valid_d
    if valid_l is None:
        valid_l = data.valid_l

    valid_preds = []
    valid_gt = []
    for i, (d, l) in enumerate(zip(valid_d, valid_l)):
        if os.path.exists("2-" + name + "_valid_%i_pred.h5" % i):
            pred = utils.h5load("2-" + name + "_valid_%i_pred.h5" % i)
        else:
            pred = model.predict_dense(d, pad_raw=False)  # (f,z,x,y)
            utils.h5save(pred, "2-" + name + "_valid_%i_pred.h5" % i)

        if betaloss:
            pred = pred[0]  # only mode
        else:
            pred = pred[1]  # only pred for class '1'
        l = l[0]  # throw away channel
        l, pred = image.center_cubes(l, pred, crop=True)
        valid_preds.append(pred)
        valid_gt.append(l)

    valid_gt = [gt > 0.5 for gt in
                valid_gt]  # binarise possibly probabilistic GT

    valid_acc, valid_area, valid_thresh = evaluate(valid_gt, valid_preds,
                                                   "1-" + name + "_valid")
    gt_flat = np.concatenate(list(map(np.ravel, valid_gt)))
    preds_flat = np.concatenate(list(map(np.ravel, valid_preds)))

    if fudgeysoft:
        valid_nll = binary_nll(rescale_fudge(preds_flat), gt_flat)
    else:
        valid_nll = binary_nll(preds_flat, gt_flat)

    print("Valid nll %.3f" % valid_nll)
    report_str += "%.3f,\t%.3f,\t%.3f,\t%.3f,\t%.3f,\t" % (
        valid_nll, valid_acc, valid_area, train_acc - valid_acc,
        train_area - valid_area)

    error_hist(gt_flat, preds_flat, "1-" + name + "_valid",
               thresh=valid_thresh)

    ris = []
    best_ris = []
    for i, (l, p) in enumerate(zip(valid_gt, valid_preds)):
        if betaloss or fudgeysoft:
            p = rescale_fudge(p)

        p_int = (p * 255).astype(np.uint8)
        ri, best_ri, seg = image.optimise_segmentation(l, p_int,
                                                       "2-" + name + "_valid_%i" % i,
                                                       n_proc=n_proc)
        best_ris.append(best_ri)
        ris.append(ri)

    ris.append(np.mean(ris))
    for ri in ris:
        report_str += "%.4f,\t" % (ri,)

    with open("0-%s-REPORT.txt" % (name,), 'w') as f:
        f.write(report_str)
