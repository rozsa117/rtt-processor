#! /usr/bin/python3
# @author: Dusan Klinec (ph4r05)
# pandas matplotlib numpy networkx graphviz scipy dill configparser coloredlogs mysqlclient requests sarge cryptography paramiko shellescape

from rtt_tools.common.rtt_db_conn import *
import configparser
import re
import math
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


def unique_justseen(iterable, key=None):
    """List unique elements, preserving order. Remember only the element just seen."""
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return list(map(next, map(lambda x: x[1], itertools.groupby(iterable, key))))


def bins(iterable, nbins=1, key=lambda x: x, ceil_bin=False):
    vals = [key(x) for x in iterable]
    min_v = min(vals)
    max_v = max(vals)
    bin_size = ((1 + max_v - min_v) / float(nbins))
    bin_size = math.ceil(bin_size) if ceil_bin else bin_size
    bins = [[] for _ in range(nbins)]
    for c in iterable:
        cv = key(c)
        cbin = int((cv - min_v) / bin_size)
        bins[cbin].append(c)
    return bins


def get_bins(iterable, nbins=1, key=lambda x: x, ceil_bin=False, full=False):
    vals = [key(x) for x in iterable]
    min_v = min(vals)
    max_v = max(vals)
    bin_size = ((1 + max_v - min_v) / float(nbins))
    bin_size = math.ceil(bin_size) if ceil_bin else bin_size
    bins = [[] for _ in range(nbins)]
    for c in iterable:
        cv = key(c)
        cbin = int((cv - min_v) / bin_size)
        bins[cbin].append(c)
    return bins if not full else (bins, bin_size, min_v, max_v, len(vals))


def get_distrib_fbins(iterable, bin_tup, non_zero=True):  # bins = bins, size, minv, maxv
    bins, size, minv, maxv, ln = bin_tup  # (idx+0.5)
    return [x for x in [(minv + (idx) * size, len(bins[idx]) / float(ln)) for idx in range(len(bins))] if
            not non_zero or x[1] > 0]


def get_distrib(iterable, nbins=100, key=lambda x: x, non_zero=True):
    inp = list(iterable)
    return get_distrib_fbins(iterable, get_bins(inp, nbins=nbins, key=key, full=True), non_zero=non_zero)


def get_bin_val(idx, bin_tup):
    return len(bin_tup[0][idx]) / float(bin_tup[4])


def get_bin_data(x, bin_tup):
    idx = binize(x, bin_tup)
    if idx is None:
        return 0, None
    return get_bin_val(idx, bin_tup), idx


def binize(x, bin_tup):  # returns bin idx where the x lies, None if out of region
    bins, size, minv, maxv, ln = bin_tup
    if x < minv or x > maxv + size:
        return None
    return int((x - minv) // size)


def binned_pmf(x, bin_tup):
    return get_bin_data(x, bin_tup)[0]


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


def median(inp, is_sorted=False):
    inp = sorted(inp) if not is_sorted else inp
    return None if not inp else (inp[len(inp) // 2])


def compress_fnc(exps, fnc):
    return [x for x in exps if fnc(x)]


def filter_experiments(exps, fnc):
    return [bool(fnc(x)) for x in exps]


def project_tests(tests_srt, bitmap):
    return [(x[0], list(itertools.compress(x[1], bitmap))) for x in tests_srt]


def get_rounds(exps):
    f2r = collections.defaultdict(lambda: [])
    for x in exps:
        f2r[x.exp_info.fnc_name].append(x.exp_info.fnc_round)
    return f2r


def get_top_rounds(exps):
    f2r = get_rounds(exps)
    return {k: max(f2r[k]) for k in f2r}


def get_low_rounds(exps):
    f2r = get_rounds(exps)
    return {k: min(f2r[k]) for k in f2r}


def get_med_rounds(exps):
    f2r = get_rounds(exps)
    return {k: median(f2r[k]) for k in f2r}


def get_exid(rev_exp, bitmap, name):  # name -> new index
    idx = rev_exp[name]
    return sum(bitmap[:idx])


def get_ex_newidx(bitmap, idx):
    return sum(bitmap[:idx])


def get_ex_byidx(bitmap, idx):  # new index -> old index
    oidx = 0
    cnt = 0
    for i, x in enumerate(bitmap):
        if x == 1:  # selected to new round
            if cnt == idx:
                return oidx
            cnt += 1
        oidx += 1
    raise ValueError('Not found')


def logdif(a, b):
    if a == 0 or b == 0: return None
    return math.log(a, 2) - math.log(b, 2)


def get_pval_partition(pval, pvalbins):
    binidx = 0
    for idx, b in enumerate(pvalbins):
        if pval < b:
            return idx
    return len(pvalbins)


def shortenBat(name):
    name = name.replace('NIST Statistical Testing Suite', 'NSTS')
    name = name.replace('Dieharder', 'D')
    name = name.replace('TestU01 Alphabit', 'U01A')
    name = name.replace('TestU01 Big Crush', 'U01BC')
    name = name.replace('TestU01 Block Alphabit', 'U01BA')
    name = name.replace('TestU01 Rabbit', 'U01R')
    name = name.replace('TestU01 Small Crush', 'U01SC')
    return name


def shortenTest(name, skip_bat=False):
    if isinstance(name, (list, tuple)):
        if skip_bat: name = name[1:]
        name = '|'.join([str(x) for x in name])
    name = shortenBat(name)
    return name


def get_battery_idx(name):
    if isinstance(name, (list,tuple)):
        name = name[0]
    if name.startswith('NIST Statistical Testing Suite'):
        return 0
    elif name.startswith('FIPS140-2'):
        return 1
    elif name.startswith('Dieharder'):
        return 2
    elif name.startswith('TestU01 Alphabit'):
        return 3
    elif name.startswith('TestU01 Big Crush'):
        return 4
    elif name.startswith('TestU01 Block Alphabit'):
        return 5
    elif name.startswith('TestU01 Rabbit'):
        return 6
    elif name.startswith('TestU01 Small Crush'):
        return 7
    elif name.lower().startswith('booltest'):
        return 8
    else:
        return -1


def get_short_bat(idx):
    x = ['NIST', 'FIPS140-2', 'Dieharder', 'U01 Alphabit', 'U01 Big Crush', 'U01 Balphabit', 'U01 Rabbit', 'U01 SCrush', 'Booltest']
    return x[idx] if idx >= 0 else '?'


def get_maximum_detections(selection, tests_srt, do_p):
    ctests = project_tests(tests_srt, selection)
    total_det = sum(selection) * len(tests_srt)
    tests_undefined = collections.defaultdict(lambda: 0)  # tname -> # of NONE in test
    total_def_det = 0

    # Fails removal & report
    for tname, tvals in ctests:
        for idx, tval in enumerate(tvals):
            if tval is None:
                eidx = get_ex_byidx(selection, idx)
                tests_undefined[tname] += 1
                # print('%s : %s' % (tname, exps[eidx].name))
            else:
                total_def_det += 1
    return total_def_det


def get_detections(ctests, alpha=1e-3):
    totals = len(ctests) * len(ctests[0][1])
    test_fails = [sum(1 for y in x[1] if y is None) for x in ctests]  # no data cases
    tests_detections = [(
        x[0],
        sum(1 for y in x[1] if y is not None and y <= alpha) / (len(x[1]) - test_fails[i]),
        sum(1 for y in x[1] if y is not None and y <= alpha),
        sum(1 for y in x[1] if y is not None and y <= alpha) + 10,
        test_fails[i],
        len(x[1])
        ) for i, x in enumerate(ctests) if (len(x[1]) - test_fails[i]) > 0]
    return tests_detections


def fill_x_data(x_data, y_data, x_data_desired, fill_fnc=lambda x: 0, sorted_add=None):
    """Augments x, y data to precisely match x_data_desired with fill function. x_data has to be sorted"""
    # Remove all not present
    n_x, n_y = [], []
    xdes_set = set(x_data_desired)
    for ix, name in enumerate(x_data):
        if name in xdes_set:
            n_x.append(x_data[ix])
            n_y.append(y_data[ix])

    # Add all missing, now is a subset
    if sorted_add is not None:
        len_cur = len(n_x)
        missing = [(idx, x) for idx, x in enumerate(x_data_desired) if x not in set(n_x)]
        for ridx, (idx, name) in enumerate(missing):
            ridx_off = ridx + len_cur if sorted_add > 0 else ridx
            n_x.insert(ridx_off, name)
            n_y.insert(ridx_off, fill_fnc((ridx_off, name)))

    else:
        off = 0
        for idx, name in enumerate(x_data_desired):
            if idx >= len(n_x) or n_x[idx] != name:
                n_x.insert(idx, name)
                n_y.insert(idx, fill_fnc((idx, name)))

    return n_x, n_y


def wrapHyphened(txt, width=20):
    res = ['']
    parts = txt.split('-')
    for ix, p in enumerate(parts):
        if len(res[-1]) + len(p) > width:
            res.append('')
        res[-1] += p + ('-' if ix < len(parts) - 1 else '')
    return res


def doChunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def filterTests(tests_srt, excl_extra=[], ret_idx=False):
    res = []
    excl = [
               'TestU01 Rabbit|smultin_MultinomialBitsOver',
               'NIST Statistical Testing Suite|Random Excursions Test',
               'NIST Statistical Testing Suite|Random Excursions Variant Test',
           ] + excl_extra

    take_idx = []
    for idx, x in enumerate(tests_srt):
        tname = x[0]
        if isinstance(tname, (list, tuple)):
            tname = '|'.join([str(y) for y in tname])
        skip = False
        for exx in excl:
            if tname.startswith(exx):
                skip = True
                break
        if not skip:
            take_idx.append(1)
            res.append(x)
        else:
            take_idx.append(0)
    return res if not ret_idx else (res, take_idx)


def sorted_perm(iterable, key=lambda x: x):
    res = sorted(zip(range(len(iterable)), iterable), key=lambda x: key(x[1]))
    return [x[1] for x in res], [x[0] for x in res]


def apply_permutation(iterable, perm):
    mp = {x: idx for idx, x in enumerate(perm)}
    return [x[1] for x in sorted(zip(range(len(iterable)), iterable), key=lambda x: mp[x[0]])]


# Coverage computation: relative coverage, one input stream - detected if detected with at least one test.
# num of all detected input streams = maximum, then measure number of detected inputs
# by subtest relative to this maximum
def comp_coverage(projected_tests, alpha=1e-5):
    res = 0
    num_tests = len(projected_tests)
    num_exps = len(projected_tests[0][1])
    for eidx in range(num_exps):
        for tidx in range(num_tests):
            c = projected_tests[tidx][1][eidx]
            if c is not None and c <= alpha:
                res += 1
                break

    return res


def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def shorten_test_name(xx):
    xx = re.sub(r'\s+test(s)?$', '', xx, flags=re.I)
    xx = re.sub(r'^dieharder\s', '', xx, flags=re.I)
    xx = re.sub(r'^diehard\s', '', xx, flags=re.I)
    xx = re.sub(r'^sts\s', '', xx, flags=re.I)
    if 'Test For Frequency Within A Block' in xx:
        xx = 'FreqInBlock'
    if 'Test for the Longest Run of Ones in a Block' in xx:
        xx = 'LongestRun1inBlock'
    if 'Non-overlapping (Aperiodic) Template Matching' in xx:
        xx = 'AperiodicTplMatch'
    if 'Overlapping (Periodic) Template Matching' in xx:
        xx = 'PeriodiTplMatch'
    if 'Discrete Fourier Transform (Spectral)' in xx:
        xx = 'Spectral'
    if 'MultinomialBitsOver' in xx:
        xx = 'MultBitsOver'
    if 'Cumulative Sum (Cusum)' in xx:
        xx = 'Cumsum'
    if 'RGB Kolmogorov-Smirnov' in xx:
        xx = 'RGB KS'
    if 'Serial Test (Generalized)' in xx:
        xx = 'Serial (Gen.)'
    if 'RGB Bit Distribution' in xx:
        xx = 'RGB BitDist'
    if 'Minimum Distance (2d Circle)' in xx:
        xx = 'MinDist (2dCirc)'
    if 'RGB Generalized Minimum Distance' in xx:
        xx = 'RGB GenMinDist'
    if 'Random Binary Matrix Rank' in xx:
        xx = 'RandBinMatrixRank'
    if "Maurer's Universal Statistical" in xx:
        xx = 'Maurer UniStat'
    if '3d Sphere (Minimum Distance)' in xx:
        xx = 'MinDist (3dSphere)'
    return xx


def process_test_x_axis(x_data, test_mapping=None, keep_battery=False):
    xx_data = []
    for xx in x_data:
        if 'booltest' in xx[0] and test_mapping:
            p = test_mapping[xx].params
            bstring = 'booltest-%s-%s-%s' if keep_battery else '%s-%s-%s'
            xx_data.append(bstring % (p.conf['m'], p.conf['deg'], p.conf['k']))

        else:
            cbt = shortenBat(xx[0])
            xx = list(xx[1:])
            if '_' in xx[0]:
                xx[0] = xx[0][xx[0].index('_') + 1:]
            xx[0] = shorten_test_name(xx[0])
            xx = '|'.join([str(x) for x in xx])
            xx_data.append(('%s|%s' % (cbt, xx)) if keep_battery else xx)

    # Need unique x_labels, otherwise cancels out (lower number than expected)
    cxx, cxxs = [], set()
    for xx in xx_data:
        while xx in cxxs: xx = ' ' + xx
        cxx.append(xx)
        cxxs.add(xx)
    return cxx


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
                 'summarized_pvals', 'summarized_passed', )

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
        self.summarized_passed = []

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

    def get_min_pval(self):
        return min(self.summarized_pvals) if self.summarized_pvals else None


class Battery:
    def __init__(self, idd, name, passed, total, alpha, exp_id, job_id=None):
        self.id = idd
        self.name = name
        self.passed = passed
        self.total = total
        self.alpha = alpha
        self.exp_id = exp_id
        self.job_id = job_id
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
        self.fnc_name = None
        self.fnc_round = None
        self.fnc_block = None
        self.key = None
        self.offset = None
        self.derivation = None

    def __repr__(self):
        return 'Einfo(id=%r, m=%r, s=%r, si=%r, osi=%r, fname=%r, fr=%r, fb=%r, k=%r, off=%r, der=%r)' % (
            self.id, self.meth, self.seed, self.size, self.osize, self.fnc_name, self.fnc_round, self.fnc_block,
            self.key, self.offset, self.derivation
        )


class Experiment:
    def __init__(self, eid, name, exp_info):
        self.id = eid
        self.name = name
        self.exp_info = exp_info  # type: ExpInfo
        self.batteries = {}  # type: dict[int, Battery]

    def __repr__(self):
        return 'Exp(%s)' % self.name


def pick_one_statistic(stats: List[Statistic], preflist=None, default_first=True) -> Optional[Statistic]:
    if len(stats) == 0:
        return None
    for st in (preflist or STATISTICAL_PREFIXES):
        for cur in stats:
            name = cur.name.lower()
            if name.startswith(st):
                return cur
    return stats[0] if default_first else None


class Loader:
    def __init__(self):
        self.args = None
        self.conn = None
        self.experiments = {}  # type: dict[int, Experiment]
        self.batteries = {}  # type: dict[int, Battery]
        self.tests = {}  # type: dict[int, Test]
        self.sids = {}  # type: dict[int, Stest]
        self.picked_stats = None
        self.add_passed = False

        self.last_bool_battery_id = 1 << 42
        self.last_bool_variant_id = 1 << 42
        self.last_bool_test_id = 1 << 42
        self.last_bool_subtest_id = 1 << 42

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

    def load_from(self, loader):
        self.experiments = loader.experiments
        self.batteries = loader.batteries
        self.tests = loader.tests
        self.sids = loader.sids
        self.picked_stats = loader.picked_stats
        self.add_passed = loader.add_passed
        self.last_bool_battery_id = loader.last_bool_battery_id
        self.last_bool_variant_id = loader.last_bool_variant_id
        self.last_bool_test_id = loader.last_bool_test_id
        self.last_bool_subtest_id = loader.last_bool_subtest_id

        self.batteries_db = loader.batteries_db
        self.tests_db = loader.tests_db
        self.usettings_db = loader.usettings_db
        self.usettings_indices_db = loader.usettings_indices_db
        self.test_par_db = loader.test_par_db
        self.test_par_indices_db = loader.test_par_indices_db
        self.stats_db = loader.stats_db

    def proc_args(self, args=None):
        parser = argparse.ArgumentParser(description='RTT result processor')
        parser.add_argument('--small', dest='small', action='store_const', const=True, default=False,
                            help='Small result set (few experiments)')
        parser.add_argument('--only-pval-cnt', dest='only_pval_cnt', action='store_const', const=True, default=False,
                            help='Load only pval counts, not actual values (faster)')
        parser.add_argument('--no-pvals', dest='no_pvals', action='store_const', const=True, default=False,
                            help='Do not load pvals')
        parser.add_argument('--exps', dest='experiments', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='Experiment numbers to load')
        parser.add_argument('--exps-id', dest='experiment_ids', nargs=argparse.ZERO_OR_MORE, default=[], type=int,
                            help='Experiment IDs numbers to load')

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
        #m = re.match(r'^EXPUNSET(?:_([\w]+?))?_seed_([\w]+?)_([\w]+?)_(?:key_([\w]*?)_off_([\w]+?)_der_([\w]+?)_*)?_([\w_-]+?)_r(-?[\w]+?)_b([\w]+?)(\.bin)?$', s)
        m = re.match(r'^TARO_DP_2020_0.(?:_([\w]+?))?_seed_([\w]+?)_([\w]+?)_(?:key_([\w]*?)_off_([\w]+?)_der_([\w]+?)_*)?_([\w_-]+?)_r(-?[\w]+?)_b([\w]+?)(\.bin)?$', s)
        if m is None:
            return ExpInfo()

        psize = self.parse_size(m.group(3))
        ei = ExpInfo(eid=0, meth=m.group(1), seed=m.group(2), osize=m.group(3), size=psize, fnc=m.group(7))
        ei.fnc_name = m.group(7)
        ei.fnc_round = int(m.group(8))
        ei.fnc_block = m.group(9)

        if m.group(4):
            ei.key = m.group(4)
            ei.offset = m.group(5)
            ei.derivation = m.group(6)

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

            if self.args.no_pvals:
                pass

            elif self.args.only_pval_cnt:
                # Pvalue counts only
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
                WHERE name LIKE '%TARO_DP_2020_01%' or name LIKE '%TARO_DP_2020_02%' or name LIKE '%TARO_DP_2020_07%'
            """)

            wanted_exps = set([int(x) for x in self.args.experiments])
            wanted_ids = set([int(x) for x in self.args.experiment_ids])

            for result in c.fetchall():
                eid, name = result
                exp_info = self.break_exp(name)
                if len(wanted_exps) > 0 and exp_info.id not in wanted_exps:
                    continue
                if len(wanted_ids) > 0 and eid not in wanted_ids:
                    continue

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
                                SELECT `id`, `name`, `passed_tests`,
                                       `total_tests`, `alpha`, `experiment_id`, 
                                       `job_id`
                                 FROM batteries 
                                WHERE experiment_id IN (%s)
                            """ % ','.join([str(x) for x in bs]))

                for result in c.fetchall():
                    self.new_battery(result[1])

                    bt = Battery(*result)
                    bt.exp = self.experiments[bt.exp_id]

                    self.batteries[bt.id] = bt
                    self.experiments[bt.exp_id].batteries[bt.id] = bt
                    bat2exp[bt.id] = bt.exp_id

            '''
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
            '''
    def load_booltest(self, js=None, fname=None, alpha=1/20000., as_subtests=True):
        if fname:
            with open(fname, 'r') as fh:
                js = json.load(fh)

        if not js:
            raise ValueError("Empty data")

        # Experiment mapping name -> exp record
        exp_name_map = {e.name: e for e in self.experiments.values()}
        sorter = lambda x: (x["data_file"], x["m"], x["deg"], x["k"])
        js = sorted(js, key=sorter)

        # Experiments matching
        for k, g in itertools.groupby(js, key=lambda x: x["data_file"]):
            name = k
            subs = list(g)

            if name not in exp_name_map:
                logger.warning("Could not find experiment %s")
                continue

            exp = exp_name_map[name]  # type: Experiment
            bid = self.last_bool_battery_id
            self.last_bool_battery_id += 1

            # Create battery
            self.new_battery('booltest')
            bt = Battery(idd=bid, name='booltest', passed=0, total=0, alpha=alpha, exp_id=exp.id)
            bt.exp = self.experiments[bt.exp_id]

            self.batteries[bt.id] = bt
            self.experiments[bt.exp_id].batteries[bt.id] = bt

            if as_subtests:
                self.load_booltest_as_subtests(bt, alpha, name, subs)
            else:
                self.load_booltest_as_tests(bt, alpha, name, subs)

    def load_booltest_as_tests(self, bt: Battery, alpha: float, exp: str, subs: list):
        self.new_param('deg')
        self.new_param('k')
        self.new_param('m')

        for idx, sub in enumerate(subs):
            tname = 'booltest-%s-%s-%s' % (sub['m'], sub['deg'], sub['k'])
            tt = Test(idd=self.last_bool_test_id, name=tname, palpha=alpha, passed=not sub['pval0_rej'],
                      test_idx=idx, battery_id=bt.id)

            tt.battery = bt
            tt.battery.tests[tt.id] = tt
            self.tests[tt.id] = tt
            self.last_bool_test_id += 1
            self.new_test(tt.name)

            # Create config
            cfg = Config({'deg': sub['deg'], 'm': sub['m'], 'k': sub['k']})
            self.new_stest_config(cfg)

            # Create variant
            tv = TVar(id=self.last_bool_variant_id, vidx=0, test_id=tt.id, settings={})
            tv.test = tt
            tv.test.variants[tv.id] = tv
            tv.settings = cfg
            self.last_bool_variant_id += 1

            # Create subtest
            stest = Stest(idd=self.last_bool_subtest_id, idx=idx, variant_id=tv.id)
            stest.params = cfg
            stest.variant = tv
            stest.variant.sub_tests[stest.id] = stest
            self.sids[stest.id] = stest
            self.last_bool_subtest_id += 1

            stat = Statistic(name='boolres-zscore', value=sub['zscore'], passed=not sub['pval0_rej'])
            stest.stats = [stat]
            self.new_stats(stat, stat.name)

            tt.passed = stat.passed
            bt.passed = tt.passed

    def load_booltest_as_subtests(self, bt: Battery, alpha: float, exp: str, subs: list):
        # Create test
        tt = Test(idd=self.last_bool_test_id, name='', palpha=alpha, passed=False,
                  test_idx=0, battery_id=bt.id)
        tt.summarized_pvals = []

        tt.battery = bt
        tt.battery.tests[tt.id] = tt
        self.tests[tt.id] = tt
        self.last_bool_test_id += 1
        self.new_test(tt.name)

        # Create variant
        tv = TVar(id=self.last_bool_variant_id, vidx=0, test_id=tt.id, settings={})
        tv.test = tt
        tv.test.variants[tv.id] = tv
        self.last_bool_variant_id += 1

        # Subtest
        num_rejects = 0
        for idx, sub in enumerate(subs):
            stest = Stest(idd=self.last_bool_subtest_id, idx=idx, variant_id=tv.id)
            stest.variant = tv
            stest.variant.sub_tests[stest.id] = stest
            self.sids[stest.id] = stest
            self.last_bool_subtest_id += 1

            cfg = Config({'deg': sub['deg'], 'm': sub['m'], 'k': sub['k']})
            self.new_stest_config(cfg)
            stest.params = cfg

            # Fake pvalue according to the precomputed pval tables as we don't have now
            # the pvalue approximation - not enough reference data.
            stat1 = Statistic(name='boolres-pval', value=1e-15 if sub['pval0_rej'] else 0.5, passed=not sub['pval0_rej'])
            stat2 = Statistic(name='boolres-zscore', value=sub['zscore'], passed=not sub['pval0_rej'])
            stest.stats = [stat1, stat2]
            self.new_stats(stat1, stat1.name)
            self.new_stats(stat2, stat2.name)
            num_rejects += 0 if stat1.passed else 1
            tt.summarized_pvals.append(stat1.value)

        tt.passed = num_rejects == 0  # num_rejects < len(subs) / 2
        bt.passed = tt.passed

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

    def pack_stat(self, stat):
        """Packs one stat value to the result. if self.add_passed is set, it returns (pvalue, passed), otherwise pvalue"""
        return stat.value if not self.add_passed else (stat.value, stat.passed)

    def pick_stats(self, stats, add_all=False, pick_one=False):
        """Picks the correct statistics to test for"""
        if len(stats) == 0:
            return []
        if len(stats) == 1:
            return [stats[0]]

        # Strategy 1: Return all statistics.
        # Makes sense if the stats are about independent parts of the test. E.g., chi-squares of different parts.
        # However if the tests are highly correlated such as Chi-Square, AD, KS test of the same thing it could
        # skew the statistics.
        if add_all:
            return stats  # [x.value for x in stats]

        # Strategy 2: Compute resulting p-value from all pvalues in collected stats.
        # Same as above, if pvalues are independent, result are better and we can compute one final pvalue.
        # WARNING: this strategy does not work well if resulting tree is unbalanced. It has to be perfectly symmetric.
        if not pick_one:
            # pvals = [x.value for x in stats]
            return stats  # [sidak_inv(min(pvals), len(pvals))]

        # Strategy 3: Pick one fixed p-value from the result. Prefer Chi-Square, then KS, then AD. First found.
        st = pick_one_statistic(stats)
        return [st]  # [st.value]

    def comp_sub_pvals(self, add_all=False, pick_one=True):
        """Computes summarized pvals"""
        self.picked_stats = collections.defaultdict(lambda: 0)

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

                    if not add_all and pick_one:
                        picked = pick_one_statistic(ss.stats)
                        self.picked_stats[picked.name] += 1

                    picked_stats = self.pick_stats(ss.stats, add_all=add_all, pick_one=pick_one)
                    picked_pvals = [x.value for x in picked_stats]
                    picked_pass = [x.passed for x in picked_stats]

                    # Sidak postprocessing.
                    # Compute resulting p-value from all pvalues in collected stats.
                    # If pvalues are independent, result are better and we can compute one final pvalue.
                    # WARNING: this strategy does not work well if resulting tree is unbalanced.
                    #          It has to be perfectly symmetric.
                    if not add_all and not pick_one:
                        picked_pvals = [sidak_inv(min(picked_pvals), len(picked_pvals))]
                        picked_pass = []

                    tt.summarized_pvals += picked_pvals
                    if self.add_passed:
                        tt.summarized_passed += picked_pass

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
            cdata['round'] = exp.exp_info.fnc_round
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

        logger.info('DONE dump_data')

        return exp_data


def main():
    l = Loader()
    return l.main()


if __name__ == "__main__":
    main()
