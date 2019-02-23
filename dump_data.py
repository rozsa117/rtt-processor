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


logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)


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
        self.params = params or None  # type: Config
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
        self.settings = settings or None  # type: Config
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
    __slots__ = ('id', 'name', 'palpha', 'passed', 'test_idx', 'battery_id', 'battery', 'variants')

    def __init__(self, idd, name, palpha, passed, test_idx, battery_id):
        self.id = idd
        self.name = name
        self.palpha = palpha
        self.passed = passed
        self.test_idx = test_idx
        self.battery_id = battery_id
        self.battery = None  # type: Battery
        self.variants = {}

    def __repr__(self):
        return 'Test(%s, battery=%s)' % (self.name, self.battery)

    def short_desc(self):
        d = [self.name]
        if self.battery:
            d += list(self.battery.short_desc())
        return d


class Battery:
    def __init__(self, idd, name, passed, total, alpha, exp_id):
        self.id = idd
        self.name = name
        self.passed = passed
        self.total = total
        self.alpha = alpha
        self.exp_id = exp_id
        self.exp = None
        self.tests = {}  # type: dict[int, Test]

    def __repr__(self):
        return 'Battery(%s, exp=%s)' % (self.name, self.exp)

    def short_desc(self):
        return [self.name, ]


class Experiment:
    def __init__(self, eid, name, parsed):
        self.id = eid
        self.name = name
        self.parsed = parsed
        self.batteries = {}  # type: dict[int, Battery]

    def __repr__(self):
        return 'Exp(%s)' % self.name


class Loader:
    def __init__(self):
        self.args = None
        self.conn = None
        self.experiments = {}  # type: dict[int, Experiment]
        self.bat2exp = {}
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

    def break_exp(self, s):
        m = re.match(r'^SECMARGINPAPER(\d)_([\w]+?)_seed_([\w]+?)_([\w]+?)__([\w_-]+?)(\.bin)?$', s)
        return m.groups() if m else None

    def queue_summary(self):
        return len(self.to_proc_test), len(self.to_proc_variant), len(self.to_proc_stest)

    def on_test_loaded(self, test):
        self.to_proc_test.append(test)
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

        self.to_proc_stest = []
        return True

    def proc_args(self):
        parser = argparse.ArgumentParser(description='RTT result processor')
        parser.add_argument('--small', dest='small', action='store_const', const=True, default=False,
                            help='Small result set (few experiments)')
        parser.add_argument('--only-pval-cnt', dest='only_pval_cnt', action='store_const', const=True, default=False,
                            help='Load only pval counts, not actual values (faster)')

        self.args = parser.parse_args()

    def main(self):
        self.proc_args()
        self.connect()

        with self.conn.cursor() as c:
            # Load all experiments
            tstart = time.time()
            logger.info("Loading all experiments")
            c.execute("""
                SELECT id, name FROM experiments 
                WHERE (name LIKE 'SECMARGINPAPER2%' OR name LIKE 'SECMARGINPAPER3%')
            """)

            batteries = []  # type: list[Battery]

            for result in c.fetchall():
                eid, name = result
                parsed = self.break_exp(name)
                self.experiments[eid] = Experiment(eid, name, parsed)

            # Load batteries for all experiments, chunked.
            eids = sorted(list(self.experiments.keys()))
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
                    batteries.append(bt)

                    self.experiments[bt.exp_id].batteries[bt.id] = bt
                    self.bat2exp[bt.id] = bt.exp_id

            # Load all tests for all batteries
            bids = sorted(list(self.bat2exp.keys()))
            bidsmap = {x.id: x for x in batteries}
            logger.info("Loading all tests, len: %s" % len(bids))
            del (batteries)

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
            logger.info('Time finished: %s' % (time.time() - tstart))
            logger.info('Num experiments: %s' % len(self.experiments))
            logger.info('Num stests: %s' % len(self.sids))
            logger.info('Queues: %s' % (self.queue_summary(),))

            res_chars = collections.defaultdict(lambda: 0)
            res_chars_tests = collections.defaultdict(lambda: set())

            for stest in self.sids.values():
                char_str = '|'.join([str(x) for x in stest.result_characteristic()])
                sdest = tuple(reversed(stest.short_desc()))

                res_chars[char_str] += 1
                res_chars_tests[char_str].add(sdest)

            res_data = collections.OrderedDict()
            res_data['res_chars'] = res_chars
            res_data['res_chars_tests'] = {k: sorted(list(res_chars_tests[k])) for k in res_chars_tests}
            json.dump(res_data, open('res_chars.json', 'w'), indent=2)

            logger.info('DONE ')


def main():
    l = Loader()
    l.main()


if __name__ == "__main__":
    main()
