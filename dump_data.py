#! /usr/bin/python3

from common.rtt_db_conn import *
import configparser
import sys
import re
import logging
import coloredlogs
import itertools
import collections
import json
import argparse
from typing import Optional, List


logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)


STATISTICAL_PREFIXES = ['chi', 'kolm', 'ander', 'ad']


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# Not used, just an example
def get_test_statistics(conn, test_id):
    with conn.cursor() as c:
        c.execute("""
            SELECT statistics.value FROM statistics
            JOIN subtests ON statistics.subtest_id=subtests.id
            JOIN variants ON subtests.variant_id=variants.id
            WHERE variants.test_id=%s
        """, (test_id, ))

        return [float(x[0]) for x in c.fetchall()]


def sidak_alpha(alpha, m):
    """
    Compute new significance level alpha for M independent tests.
    More tests -> unexpected events occur more often thus alpha has to be adjusted.
    Overall test battery fails if min(pvals) < new_alpha.
    """
    return 1 - (1 - alpha)**(1./m)


def sidak_inv(alpha, m):
    """
    Inverse transformation of sidak_alpha function.
    Used to compute final p-value of M independent tests if while preserving the
    same significance level for the resulting p-value.
    """
    return 1 - (1 - alpha)**m


def test_sidak():
    alpha = .01
    nvals = [.07, .02, .002007, 0.98, 0.55]
    ns_ex = sidak_alpha(alpha, len(nvals))
    nsicp = sidak_inv(min(nvals), len(nvals))

    print('alpha: %s' % alpha)
    print('min p: %s' % min(nvals))
    print('sh  t: %s, %s' % (ns_ex, min(nvals) < ns_ex))
    print('shi p: %s, %s' % (nsicp, nsicp < alpha))

    for alpha in [.01, .05]:
        for new_value in [.001, .002, .002007, .002008, .002009, .003, .004, .01]:
            nvals = [.07, .02, 0.98, 0.55] + [new_value]
            ns_ex = sidak_alpha(alpha, len(nvals))
            nsicp = sidak_inv(min(nvals), len(nvals))
            if (min(nvals) < ns_ex) != (nsicp < alpha):
                print('alpha: %s' % alpha)
                print('min p: %s' % min(nvals))
                print('sh  t: %s, %s' % (ns_ex, min(nvals) < ns_ex))
                print('shi p: %s, %s' % (nsicp, nsicp < alpha))
                raise ValueError('ERROR')


def merge_pvals(pvals, batch=2):
    """
    Merging pvals with Sidak.

    Note that the merging tree has to be symmetric, otherwise the computation on pvalues is not correct.
    Note: 1-(1-(1-(1-x)^3))^2 == 1-((1-x)^3)^2 == 1-(1-x)^6.
    Example: 12 nodes, binary tree: [12] -> [2,2,2,2,2,2] -> [2,2,2]. So far it is symmetric.
    The next layer of merge is problematic as we merge [2,2] and [2] to two p-values.
    If a minimum is from [2,2] (L) it is a different expression as from [2] R as the lists
    have different lengths. P-value from [2] would increase in significance level compared to Ls on this new layer
    and this it has to be corrected.
    On the other hand, the L minimum has to be corrected as well as it came from
    list of the length 3. We want to obtain 2 p-values which can be merged as if they were equal (exponent 2).
    Thus the exponent on the [2,2] and [2] layer will be 3/2 as we had 3 p-values in total and we are producing 2.
    """
    if len(pvals) <= 1:
        return pvals

    batch = min(max(2, batch), len(pvals))  # norm batch size
    parts = list(chunks(pvals, batch))
    exponent = len(pvals) / len(parts)
    npvals = []
    for p in parts:
        pi = sidak_inv(min(p), exponent)
        npvals.append(pi)
    return merge_pvals(npvals, batch)


def is_statistical_test(name):
    return name.lower() in STATISTICAL_PREFIXES


class Config:
    def __init__(self, conf=None):
        self.conf = conf or {}
        self.id = None

    def hashable(self):
        """Hashable representation of the configuration, with values"""
        return tuple((k, self.conf[k]) for k in sorted(self.conf.keys()))

    def keys_tuple(self):
        """Hashable representation of configuration keys"""
        return tuple(sorted(self.conf.keys()))

    def values_tuple(self):
        """Hashable representation of configuration values"""
        return tuple([self.conf[x] for x in sorted(self.conf.keys())])

    @property
    def cfg(self):
        return self.conf

    def __repr__(self):
        return '{%s}' % (', '.join(['%s: %s' % (x, self.conf[x]) for x in sorted(self.conf.keys())]))


class Statistic:
    __slots__ = ('name', 'nameidx', 'value', 'passed')

    def __init__(self, name, value, passed):
        self.name = name
        self.nameidx = None
        self.value = value
        self.passed = passed

    def __repr__(self):
        return '%s:%s:%s' % (self.name if self.name else self.nameidx, self.value, self.passed)


class Stest:
    __slots__ = ('id', 'idx', 'variant_id', 'params', 'pvals', 'npvals', 'stats', 'variant')

    def __init__(self, idd, idx, variant_id, params=None):
        self.id = idd
        self.idx = idx  # subtest index in the variant
        self.variant_id = variant_id
        self.params = params or Config()  # type: Config
        self.pvals = []  # type: list[float]
        self.stats = []  # type: list[Statistic]
        self.variant = None  # type: TVar
        self.npvals = 0

    def set_pvals(self, pvals):
        self.pvals = pvals
        self.npvals = len(pvals)

    def set_pvals_cnt(self, cnt):
        self.npvals = cnt

    def __repr__(self):
        return 'Subtest(%s, %s, lenpvals=%s, stats=%s, variant=%s)' % (self.idx, self.params, self.npvals, self.stats, self.variant)

    def result_characteristic(self):
        return tuple([self.npvals > 0] + sorted([x.name for x in self.stats]))

    def short_desc(self):
        d = [self.idx]
        if self.variant:
            d += list(self.variant.short_desc())
        return d


class TVar:
    __slots__ = ('id', 'vidx', 'test_id', 'settings', 'sub_tests', 'test')

    def __init__(self, id, vidx, test_id, settings=None):
        self.id = id
        self.vidx = vidx  # variant_index in the test set
        self.test_id = test_id
        self.settings = settings or Config()  # type: Config
        self.sub_tests = {}  # type: dict[int, Stest]
        self.test = None  # type: Test

    def __repr__(self):
        return 'Variant(%s, %s, test=%s)' % (self.vidx, self.settings, self.test)

    def short_desc(self):
        d = [self.vidx]
        if self.test:
            d += list(self.test.short_desc())
        return d


class Test:
    __slots__ = ('id', 'name', 'palpha', 'passed', 'test_idx', 'battery_id', 'battery', 'variants',
                 'summarized_pvals', )

    def __init__(self, idd, name, palpha, passed, test_idx, battery_id):
        self.id = idd
        self.name = name
        self.palpha = palpha
        self.passed = passed
        self.test_idx = test_idx
        self.battery_id = battery_id
        self.battery = None  # type: Battery
        self.variants = {}  # type: dict[int, TVar]
        self.summarized_pvals = []

    def __repr__(self):
        return 'Test(%s, battery=%s)' % (self.name, self.battery)

    def short_desc(self):
        d = [self.name]
        if self.battery:
            d += list(self.battery.short_desc())
        return d

    def shidak_alpha(self, alpha):
        return sidak_alpha(alpha, len(self.summarized_pvals)) if self.summarized_pvals else None

    def get_single_pval(self):
        return sidak_inv(min(self.summarized_pvals), len(self.summarized_pvals)) if self.summarized_pvals else None


class Battery:
    def __init__(self, idd, name, passed, total, alpha, exp_id):
        self.id = idd
        self.name = name
        self.passed = passed
        self.total = total
        self.alpha = alpha
        self.exp_id = exp_id
        self.exp = None  # type: Experiment
        self.tests = {}  # type: dict[int, Test]

    def __repr__(self):
        return 'Battery(%s, exp=%s)' % (self.name, self.exp)

    def short_desc(self):
        return [self.name, ]


class ExpInfo:
    def __init__(self, eid=None, meth=None, seed=None, osize=None, size=None, fnc=None):
        self.id = eid
        self.meth = meth
        self.seed = seed
        self.osize = osize
        self.size = size
        self.fnc = fnc


class Experiment:
    def __init__(self, eid, name, exp_info):
        self.id = eid
        self.name = name
        self.exp_info = exp_info  # type: ExpInfo
        self.batteries = {}  # type: dict[int, Battery]

    def __repr__(self):
        return 'Exp(%s)' % self.name


def pick_one_statistic(stats: List[Statistic]) -> Optional[Statistic]:
    if len(stats) == 0:
        return None
    for st in STATISTICAL_PREFIXES:
        for cur in stats:
            name = cur.name.lower()
            if name.startswith(st):
                return cur
    return stats[0]


class Loader:
    def __init__(self):
        self.args = None
        self.conn = None
        self.experiments = {}  # type: dict[int, Experiment]
        self.batteries = {}  # type: dict[int, Battery]
        self.tests = {}  # type: dict[int, Test]
        self.sids = {}  # type: dict[int, Stest]

        self.batteries_db = {}
        self.tests_db = {}
        self.usettings_db = {}
        self.usettings_indices_db = {}
        self.test_par_db = {}
        self.test_par_indices_db = {}
        self.stats_db = {}

        self.to_proc_test = []  # type: list[Test]
        self.to_proc_variant = []  # type: list[TVar]
        self.to_proc_stest = []  # type: list[Stest]

    def proc_args(self, args=None):
        parser = argparse.ArgumentParser(description='RTT result processor')
        parser.add_argument('--small', dest='small', action='store_const', const=True, default=False,
                            help='Small result set (few experiments)')
        parser.add_argument('--only-pval-cnt', dest='only_pval_cnt', action='store_const', const=True, default=False,
                            help='Load only pval counts, not actual values (faster)')

        self.args, unparsed = parser.parse_known_args()
        logger.debug("Unparsed: %s" % unparsed)

        # args override
        if not args:
            return

        for k in args:
            setattr(self.args, k, args[k])

    def connect(self):
        cfg = configparser.ConfigParser()
        cfg.read("config.ini")
        self.conn = create_mysql_db_conn(cfg)

    def new_battery(self, b):
        if b not in self.batteries_db:
            self.batteries_db[b] = len(self.batteries_db)

    def new_test(self, t):
        if t not in self.tests_db:
            self.tests_db[t] = len(self.tests_db)

    def new_usetting(self, t):
        if t not in self.usettings_db:
            self.usettings_db[t] = len(self.usettings_db)

    def new_usetting_config(self, cfg: Config):
        h = cfg.hashable()

        if h not in self.usettings_indices_db:
            cfg.id = len(self.usettings_indices_db)
            self.usettings_indices_db[h] = cfg
        else:
            cfg.id = self.usettings_indices_db[h].id

    def new_stest_config(self, cfg: Config):
        h = cfg.hashable()

        if h not in self.test_par_indices_db:
            cfg.id = len(self.test_par_indices_db)
            self.test_par_indices_db[h] = cfg
        else:
            cfg.id = self.test_par_indices_db[h].id

    def new_param(self, t):
        if t not in self.test_par_db:
            self.test_par_db[t] = len(self.test_par_db)

    def new_stats(self, s: Statistic, name):
        if name not in self.stats_db:
            self.stats_db[name] = len(self.stats_db)
        s.nameidx = self.stats_db[name]
        return s

    def parse_size(self, s):
        m = re.match(r'^(\d+)(?:([kmgt])(i)?)?b$', s.lower(), re.I)
        if m is None:
            return m

        tbl = {None: 0, 'k': 1, 'm': 2, 'g': 3, 't': 4}
        base = 1000 if m.group(3) is None else 1024
        return int(m.group(1)) * pow(base, tbl[m.group(2)])

    def break_exp(self, s):
        m = re.match(r'^SECMARGINPAPER(\d)_([\w]+?)_seed_([\w]+?)_([\w]+?)__([\w_-]+?)(\.bin)?$', s)
        if m is None:
            return ExpInfo()

        psize = self.parse_size(m.group(4))
        ei = ExpInfo(eid=m.group(1), meth=m.group(2), seed=m.group(3), osize=m.group(4), size=psize, fnc=m.group(5))
        return ei

    def queue_summary(self):
        return len(self.to_proc_test), len(self.to_proc_variant), len(self.to_proc_stest)

    def on_test_loaded(self, test):
        self.to_proc_test.append(test)
        self.tests[test.id] = test
        self.process_test()

    def process_test(self, force=False):
        if len(self.to_proc_test) == 0:
            return False
        if len(self.to_proc_test) < 1000 and not force:
            return False

        ids = sorted(list([x.id for x in self.to_proc_test]))
        idmap = {x.id: x for x in self.to_proc_test}  # type: dict[int, Test]
        logger.info("Loading all variants params, len: %s" % len(ids))

        with self.conn.cursor() as c:
            c.execute("""
                        SELECT * FROM variants 
                        WHERE test_id IN (%s)
                      """ % ','.join([str(x) for x in ids]))

            for r in c.fetchall():
                tv = TVar(*r)
                tv.test = idmap[tv.test_id]
                tv.test.variants[tv.id] = tv
                self.on_variant_loaded(tv)

        self.to_proc_test = []
        return True

    def on_variant_loaded(self, variant):
        self.to_proc_variant.append(variant)
        self.process_variant()

    def process_variant(self, force=False):
        if len(self.to_proc_variant) == 0:
            return False
        if len(self.to_proc_variant) < 2000 and not force:
            return False

        vids = sorted(list([x.id for x in self.to_proc_variant]))
        vidmap = {x.id: x for x in self.to_proc_variant}
        logger.info("Loading all variant params, len: %s" % len(vids))

        with self.conn.cursor() as c:
            # All user_settings
            c.execute("""
                        SELECT * FROM user_settings 
                        WHERE variant_id IN (%s) ORDER BY variant_id
                      """ % ','.join([str(x) for x in vids]))

            for k, g in itertools.groupby(c.fetchall(), lambda x: x[3]):
                cfg = {}
                for cc in g:
                    cfg[cc[1]] = cc[2]
                    self.new_usetting(cc[1])

                cfg = Config(cfg)
                self.new_usetting_config(cfg)
                vidmap[k].settings = cfg

            # All subtests
            c.execute("""
                        SELECT * FROM subtests 
                        WHERE variant_id IN (%s) ORDER BY variant_id
                      """ % ','.join([str(x) for x in vids]))

            for k, g in itertools.groupby(c.fetchall(), lambda x: x[2]):  # variant grouping
                for cc in g:
                    stest = Stest(*cc)
                    stest.variant = vidmap[k]
                    stest.variant.sub_tests[stest.id] = stest
                    self.on_stest_loaded(stest)

        self.to_proc_variant = []
        return True

    def on_stest_loaded(self, stest):
        self.sids[stest.id] = stest
        self.to_proc_stest.append(stest)
        self.process_stest()

    def process_stest(self, force=False):
        if len(self.to_proc_stest) == 0:
            return False
        if len(self.to_proc_stest) < 10000 and not force:
            return False

        sids = sorted(list([x.id for x in self.to_proc_stest]))
        sidmap = {x.id: x for x in self.to_proc_stest}  # type: dict[int, Stest]
        logger.info("Loading all subtest params, len: %s" % len(sids))

        with self.conn.cursor() as c:
            # All user_settings
            c.execute("""
                        SELECT * FROM test_parameters 
                        WHERE subtest_id IN (%s) ORDER BY subtest_id
                      """ % ','.join([str(x) for x in sids]))

            for k, g in itertools.groupby(c.fetchall(), lambda x: x[3]):
                cfg = {}
                for cc in g:
                    cfg[cc[1]] = cc[2]
                    self.new_param(cc[1])

                cfg = Config(cfg)
                sidmap[k].params = cfg
                self.new_stest_config(cfg)

            # Pvalue counts only
            if self.args.only_pval_cnt:
                logger.info("Loading all subtest pval counts, len: %s" % len(sids))
                c.execute("""
                            SELECT COUNT(*), `subtest_id` FROM p_values
                            WHERE subtest_id IN (%s) GROUP BY `subtest_id` ORDER BY subtest_id
                          """ % ','.join([str(x) for x in sids]))

                for cnt, k in c.fetchall():
                    sidmap[k].set_pvals_cnt(cnt)

            else:
                # All pvalues (heavy)
                logger.info("Loading all subtest pvalues, len: %s" % len(sids))
                c.execute("""
                            SELECT `value`, `subtest_id` FROM p_values
                            WHERE subtest_id IN (%s) ORDER BY subtest_id
                          """ % ','.join([str(x) for x in sids]))

                for k, g in itertools.groupby(c.fetchall(), lambda x: x[1]):
                    sidmap[k].set_pvals([x[0] for x in g])

            # All statistics
            logger.info("Loading all subtest stats, len: %s" % len(sids))
            c.execute("""
                        SELECT `name`, `value`, `result`, `subtest_id` FROM statistics 
                        WHERE subtest_id IN (%s) ORDER BY subtest_id
                      """ % ','.join([str(x) for x in sids]))

            for k, g in itertools.groupby(c.fetchall(), lambda x: x[3]):
                for st in g:
                    stat = Statistic(st[0], st[1], st[2] == 'passed')
                    self.new_stats(stat, st[0])
                    sidmap[k].stats.append(stat)

                # Sort stats according to the name
                sidmap[k].stats.sort(key=lambda x: x.name)

        self.to_proc_stest = []
        return True

    def load_data(self):
        with self.conn.cursor() as c:
            # Load all experiments
            logger.info("Loading all experiments")
            c.execute("""
                SELECT id, name FROM experiments 
                WHERE (name LIKE 'SECMARGINPAPER2%' OR name LIKE 'SECMARGINPAPER3%')
            """)

            for result in c.fetchall():
                eid, name = result
                exp_info = self.break_exp(name)
                self.experiments[eid] = Experiment(eid, name, exp_info)

            # Load batteries for all experiments, chunked.
            eids = sorted(list(self.experiments.keys()))
            bat2exp = {}
            logger.info("Number of all experiments: %s" % len(eids))

            if self.args.small:
                eids = eids[0:2]

            logger.info("Loading all batteries, len: %s" % len(eids))

            for bs in chunks(eids, 10):
                c.execute("""
                                SELECT * FROM batteries 
                                WHERE experiment_id IN (%s)
                            """ % ','.join([str(x) for x in bs]))

                for result in c.fetchall():
                    self.new_battery(result[1])

                    bt = Battery(*result)
                    bt.exp = self.experiments[bt.exp_id]

                    self.batteries[bt.id] = bt
                    self.experiments[bt.exp_id].batteries[bt.id] = bt
                    bat2exp[bt.id] = bt.exp_id

            # Load all tests for all batteries
            bids = sorted(list(bat2exp.keys()))
            bidsmap = {x.id: x for x in self.batteries.values()}
            logger.info("Loading all tests, len: %s" % len(bids))

            for bs in chunks(bids, 20):
                c.execute("""
                                SELECT * FROM tests 
                                WHERE battery_id IN (%s)
                            """ % ','.join([str(x) for x in bs]))

                for r in c.fetchall():
                    self.new_test(r[1])
                    tt = Test(r[0], r[1], r[2], r[3] == 'passed', r[4], r[5])
                    tt.battery = bidsmap[tt.battery_id]
                    tt.battery.tests[tt.id] = tt
                    self.new_test(tt.name)
                    self.on_test_loaded(tt)

            # Triggered processing now, too many info
            self.process_test(True)
            self.process_variant(True)
            self.process_stest(True)

    def init(self, args=None):
        self.proc_args(args)
        self.connect()

    def load(self, args=None):
        self.init(args)

        tstart = time.time()
        self.load_data()
        logger.info('Time finished: %s' % (time.time() - tstart))
        logger.info('Num experiments: %s' % len(self.experiments))
        logger.info('Num batteries: %s' % len(self.batteries))
        logger.info('Num tests: %s' % len(self.tests))
        logger.info('Num stests: %s' % len(self.sids))
        logger.info('Queues: %s' % (self.queue_summary(),))

    def process(self):
        """Computes summarized pvals"""
        self.comp_sub_pvals()

    def pick_stats(self, stats, add_all=False, pick_one=False):
        """Picks the correct statistics to test for"""
        if len(stats) == 0:
            return []
        if len(stats) == 1:
            return [stats[0].value]

        # Strategy 1: Return all statistics.
        # Makes sense if the stats are about independent parts of the test. E.g., chi-squares of different parts.
        # However if the tests are highly correlated such as Chi-Square, AD, KS test of the same thing it could
        # skew the statistics.
        if add_all:
            return [x.value for x in stats]

        # Strategy 2: Compute resulting p-value from all pvalues in collected stats.
        # Same as above, if pvalues are independent, result are better and we can compute one final pvalue.
        # WARNING: this strategy does not work well if resulting tree is unbalanced. It has to be perfectly symmetric.
        if not pick_one:
            pvals = [x.value for x in stats]
            return [sidak_inv(min(pvals), len(pvals))]

        # Strategy 3: Pick one fixed p-value from the result. Prefer Chi-Square, then KS, then AD. First found.
        st = pick_one_statistic(stats)
        return [st.value]

    def comp_sub_pvals(self):
        """Computes summarized pvals"""
        for tt in self.tests.values():
            tt_id = '|'.join(reversed(tt.short_desc()))

            for vv in tt.variants.values():
                # cfs = '|'.join([str(x) for x in vv.settings.keys_tuple()])
                cfv = '|'.join([str(x) for x in vv.settings.values_tuple()])

                for ss in vv.sub_tests.values():
                    # tfs = '|'.join([str(x) for x in ss.params.keys_tuple()])
                    tfv = '|'.join([str(x) for x in ss.params.values_tuple()])
                    if len(ss.stats) == 0:
                        logger.debug('Null statistics for test %s:%s:%s' % (tt_id, cfv, tfv))
                        continue

                    tt.summarized_pvals += self.pick_stats(ss.stats, add_all=False, pick_one=True)

    def comp_exp_data(self):
        exp_data = collections.OrderedDict()
        for exp in self.experiments.values():
            cdata = collections.OrderedDict()
            cdata['id'] = exp.id
            cdata['name'] = exp.name
            cdata['meth'] = exp.exp_info.meth
            cdata['seed'] = exp.exp_info.seed
            cdata['size'] = exp.exp_info.size
            cdata['osize'] = exp.exp_info.osize
            cdata['fnc'] = exp.exp_info.fnc
            cdata['batteries'] = []

            for bt in exp.batteries.values():
                bdata = collections.OrderedDict()
                bdata['id'] = bt.id
                bdata['name'] = bt.name
                bdata['passed'] = bt.passed
                bdata['alpha'] = bt.alpha
                bdata['total'] = bt.total
                bdata['tests'] = []
                cdata['batteries'].append(bdata)

                for tt in bt.tests.values():
                    tdata = collections.OrderedDict()
                    bdata['tests'].append(tdata)
                    tdata['id'] = tt.id
                    tdata['name'] = tt.name
                    tdata['passed'] = tt.passed
                    tdata['palpha'] = tt.palpha
                    tdata['variants'] = []

                    for vt in tt.variants.values():
                        vdata = collections.OrderedDict()
                        tdata['variants'].append(vdata)
                        vdata['id'] = vt.id
                        vdata['idx'] = vt.vidx
                        vdata['settings'] = vt.settings.conf
                        vdata['subtests'] = []

                        for st in vt.sub_tests.values():
                            sdata = collections.OrderedDict()
                            vdata['subtests'].append(sdata)
                            sdata['id'] = st.id
                            sdata['idx'] = st.idx
                            sdata['params'] = st.params.conf
                            sdata['res_char'] = st.result_characteristic()

            # experiment
            exp_data[exp.id] = cdata
        return exp_data

    def main(self, args=None):
        self.load(args)

        res_chars = collections.defaultdict(lambda: 0)
        res_chars_tests = collections.defaultdict(lambda: set())

        for stest in self.sids.values():
            char_str = '|'.join([str(x) for x in stest.result_characteristic()])
            sdest = tuple(reversed(stest.short_desc()))

            res_chars[char_str] += 1
            res_chars_tests[char_str].add(sdest)

        exp_data = self.comp_exp_data()

        # Data sizes -> tests -> test_config -> counts
        test_configs = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: 0
                )))

        # Data sizes -> tests -> test_config -> config_data -> counts
        test_configs_val = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: collections.defaultdict(
                        lambda: 0
                ))))

        for tt in self.tests.values():
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

                    test_configs[size][tt_id][tcfg] += 1
                    test_configs_val[size][tt_id][tcfg][tcfg_val] += 1

        res_data = collections.OrderedDict()
        res_data['res_chars'] = res_chars
        res_data['res_chars_tests'] = {k: sorted(list(res_chars_tests[k])) for k in res_chars_tests}
        res_data['test_configs'] = test_configs
        res_data['test_configs_val'] = test_configs_val
        json.dump(res_data, open('res_chars.json', 'w'), indent=2)

        res_data['exp_data'] = exp_data
        json.dump(res_data, open('res_chars_full.json', 'w'), indent=2)

        logger.info('DONE ')


def main():
    l = Loader()
    l.main()


if __name__ == "__main__":
    main()
