from rtt_tools import dump_data
from rtt_tools.dump_data import *  # pussy died because of this wildcard import

import collections
import json
import re
import os
import math
import random
import itertools

logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.INFO, use_chroot=False)


def process_syso(tests):
    """
    Export to JSON in a format drafted with Syso
    :param tests:
    :return:
    """
    res = []
    resmap = {}

    for ttidx, tt in enumerate(tests.values()):
        tt_id = '|'.join(reversed(tt.short_desc()))

        for vvidx, vv in enumerate(tt.variants.values()):  # variant
            cfs = '|'.join([str(x) for x in vv.settings.keys_tuple()])
            cfv = '|'.join([str(x) for x in vv.settings.values_tuple()])

            tmpsubs = []
            for ssidx, ss in enumerate(vv.sub_tests.values()):  # subtest
                tfv = '|'.join([str(x) for x in ss.params.values_tuple()])
                tmpsubs.append((tfv, ss))

            subs_acc = []
            sorter = lambda x: x[0]
            for k, g in itertools.groupby(sorted(tmpsubs, key=sorter), sorter):
                subs_acc.append([x[1] for x in g])

            for skidx, gs in enumerate(subs_acc):
                ss = gs[0]
                tfs = '|'.join([str(x) for x in ss.params.keys_tuple()])
                tfv = '|'.join([str(x) for x in ss.params.values_tuple()])

                tsubs = []
                cobj = collections.OrderedDict([
                    ('test', tt_id),
                    ('subtest', '%s' % tfv),
                    ('subtest_type', tfs),
                    ('variant', cfv),
                    ('variant_type', cfs),
                    ('variant_id', vv.id),
                    ('exid', tt.battery.exp.id),
                    ('data_type', tt.battery.exp.name),
                    ('subs', tsubs),
                ])

                for ssidx, ss in enumerate(gs):
                    cstats = [collections.OrderedDict([
                        ('name', _st.name),
                        ('value', _st.value),
                        ('pass', _st.passed),
                    ]) for _st in ss.stats]

                    csub = collections.OrderedDict([
                        ('sid', ss.id),
                        ('idx', ss.idx),
                        ('pvals', ss.pvals),
                        ('stats', cstats)
                    ])
                    tsubs.append(csub)

                res.append(cobj)

        # if ttidx > 1:
        #    break
    return res


def process_syso2(loader, exps, rev_exp):
    """
    Another JSON export format
    :param loader:
    :param exps:
    :param rev_exp:
    :return:
    """
    tests_mult = collections.defaultdict(lambda: [None] * len(exps))
    for tt in loader.tests.values():
        tt_id = '|'.join(reversed(tt.short_desc()))
        tests_mult[tt_id][rev_exp[tt.battery.exp.name]] = list(zip(tt.summarized_pvals, tt.summarized_passed))

    sys_grouper = lambda x: (x.size, x.meth, x.fnc_name, x.fnc_round)
    exps_sys = sorted(exps, key=lambda x: sys_grouper(x.exp_info))

    all_pvals = []
    tests_to_use = tests_mult
    tests_sys = collections.defaultdict(lambda: collections.defaultdict(lambda: list))
    for ti, tt in enumerate(tests_to_use):

        # Sort so the experiments that should be grouped are next to each other. Only for valid pvalues (test finished)
        cur = sorted([(eidx, pval) for eidx, pval in enumerate(tests_to_use[tt]) if pval is not None],
                     key=lambda x: sys_grouper(exps[x[0]].exp_info))

        # Group by the experiments with same size-meth-fnc-round
        for k, g in itertools.groupby(cur, key=lambda x: sys_grouper(exps[x[0]].exp_info)):
            g = list(g)
            ckey = '|'.join([str(x) for x in k])
            tests_sys[tt][ckey] = [x[1] for x in g]
            for l1 in [x[1] for x in g]:
                for l2 in l1:
                    all_pvals.append(l2[0])

    return tests_sys


class Exporter:
    def __init__(self, loader=None):
        self.loader = loader
        self.experiment_ids = None
        self.experiments = None

        self.test_configs = None
        self.test_configs_val = None
        self.flat_configs_types = None
        self.flat_configs = None
        self.test_configs_var = None
        self.all_subs = None
        self.fname = None
        self.res = None

        self.exps = None
        self.rev_exp = None

    def load(self, args):
        """Example: {'no_pvals': False, 'only_pval_cnt': False, 'experiments': exp_id_list}"""
        if 'experiments' in args:
            self.experiments = args['experiments']
        elif 'experiment_ids' in args:
            self.experiment_ids = args['experiment_ids']
        else:
            logger.warning('Unexpected loading argument, will continue, lets see what happens...')
        self.loader.load(args)

    def load_data(self):
        loader = self.loader
        logger.info("Processing the results")
        loader.add_passed = True
        loader.comp_sub_pvals(add_all=True, pick_one=False)

    def process_categories(self):
        # Categorization
        # Data sizes -> tests -> test_config -> counts
        self.test_configs = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: 0
                )))

        # Data sizes -> tests -> test_config -> config_data -> counts
        self.test_configs_val = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: collections.defaultdict(
                        lambda: 0
                    ))))

        # data size -> test_flat -> counts
        self.flat_configs_types = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: 0
            ))

        # data size -> test_flat -> counts
        self.flat_configs = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: 0
            ))

        # Data sizes -> tests -> test_config+variantval -> counts
        self.test_configs_var = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: 0
                )))

        self.all_subs = []
        for tt in self.loader.tests.values():
            exp = tt.battery.exp
            size = exp.exp_info.size
            tt_id = '|'.join(reversed(tt.short_desc()))

            for vv in tt.variants.values():
                cfs = '|'.join([str(x) for x in vv.settings.keys_tuple()])
                cfv = '|'.join([str(x) for x in vv.settings.values_tuple()])

                for ss in vv.sub_tests.values():
                    tfs = '|'.join([str(x) for x in ss.params.keys_tuple()])
                    tfv = '|'.join([str(x) for x in ss.params.values_tuple()])
                    tcfg = '{%s}{%s}' % (cfs, tfs)
                    tcfg_val = '{%s}{%s}' % (cfv, tfv)
                    type_val = '[%s][%s]' % (tt_id, tcfg)
                    var_val = '{%s}{%s}' % (cfs, cfv)
                    full_val = '[%s][%s][%s]' % (tt_id, tcfg, tcfg_val)

                    self.test_configs[size][tt_id][tcfg] += 1
                    self.test_configs['ALL'][tt_id][tcfg] += 1

                    self.test_configs_var[size][tt_id][var_val] += 1
                    self.test_configs_var['ALL'][tt_id][var_val] += 1

                    self.test_configs_val[size][tt_id][tcfg][tcfg_val] += 1
                    self.test_configs_val['ALL'][tt_id][tcfg][tcfg_val] += 1

                    self.flat_configs[size][full_val] += 1
                    self.flat_configs['ALL'][full_val] += 1
                    self.flat_configs_types[size][type_val] += 1
                    self.flat_configs_types['ALL'][type_val] += 1

                    if len(ss.pvals) > 0:
                        self.all_subs.append(ss)

    def process(self):

        # Processing
        # Test analysis and scoring
        self.exps = list(self.loader.experiments.values())
        self.exps.sort(key=lambda x: (
        x.exp_info.size, x.exp_info.meth, x.exp_info.seed, x.exp_info.fnc_name, x.exp_info.fnc_round))
        exps_grouper = lambda x: (x.exp_info.size, x.exp_info.meth, x.exp_info.fnc_name,
                                  x.exp_info.fnc_round)  # aggregate different runs with different SEEDs
        self.rev_exp = {x.name: i for i, x in enumerate(self.exps)}

        # Iterate over test
        # test[name] = [pvals] per experiment, on a fixed position. None if not present.
        tests = collections.defaultdict(lambda: [None] * len(self.exps))
        test_ids_counts = collections.defaultdict(lambda: 0)
        for tt in self.loader.tests.values():
            tt_id = '|'.join(reversed(tt.short_desc()))
            tests[tt_id][self.rev_exp[tt.battery.exp.name]] = tt.get_single_pval()
            # print(tt.get_single_pval(), tt.shidak_alpha(0.10), tt.summarized_pvals)

        # Sort tests, so we have defined ordering
        tests_srt = [(k, tests[k]) for k in tests]
        tests_srt.sort(key=lambda x: x[0])
        return tests, tests_srt

    def export(self, base_path='/tmp', fname=None):
        if fname is None:
            if self.experiments:
                fname = 'rtt-results-dump-%s.json' % ('-'.join([str(x) for x in self.experiments]))
            elif self.experiment_ids:
                fname = 'rtt-results-dump-EID%s.json' % ('-'.join([str(x) for x in self.experiments]))
            else:
                raise ValueError('Could not auto-determine result fname, please provide fname argument')

        loader = self.loader
        self.load_data()

        # Processing
        logger.info("Exporting to the required data format")
        res = process_syso(loader.tests)
        self.res = res

        logger.info("Dumping to json to: %s" % fname)
        rpath = os.path.join(base_path, fname)
        self.fname = rpath

        json.dump(res, open(rpath, 'w+'), indent=2)

