import sys
from scipy import misc

if sys.version_info >= (3, 2):
    from functools import lru_cache
else:
    from repoze.lru import lru_cache

if hasattr(scipy.misc, 'comb'):
    scipy_comb = scipy.misc.comb
else:
    import scipy.special

    scipy_comb = scipy.special.comb


@lru_cache(maxsize=1024)
def comb(n, k, exact=False):
    return scipy_comb(n, k, exact=exact)


def rank(s, n):
    """
    Returns index of the combination s in (N,K)
    https://computationalcombinatorics.wordpress.com/2012/09/10/ranking-and-unranking-of-combinations-and-permutations/
    :param s:
    :return:
    """
    k = len(s)
    r = 0
    for i in range(0, k):
        for v in range(s[i - 1] + 1 if i > 0 else 0, s[i]):
            r += comb_cached(n - v - 1, k - i - 1)
    return r


@lru_cache(maxsize=8192)
def unrank(i, n, k):
    """
    returns the i-th combination of k numbers chosen from 0,2,...,n-1, indexing from 0
    """
    c = []
    r = i + 0
    j = 0
    for s in range(1, k + 1):
        cs = j + 1

        while True:
            decr = comb(n - cs, k - s)
            if r > 0 and decr == 0:
                raise ValueError('Invalid index')
            if r - decr >= 0:
                r -= decr
                cs += 1
            else:
                break

        c.append(cs - 1)
        j = cs
    return c


def partition_space(hw, bits, partitions, debug=0):
    comb_space = int(comb(bits, hw))
    total_size = bits * comb_space
    total_size_mb = total_size / 8 / 1024 / 1024

    chunk_size = int(comb_space // partitions)
    if debug:
        print('HW: %s, Block size: %s bits\n  Total combinations: %s = %.2f MB stream size'
              % (hw, bits, comb_space, total_size_mb))
        print('\nPartitioning to %s parts, partition size: %s, sequence size: %.2f MB\n'
              % (partitions, chunk_size, bits * chunk_size / 8 / 1024 / 1024))

    res = []
    for i in range(partitions):
        offset = i * chunk_size
        state = unrank(offset, bits, hw)
        res.append(state)
        if debug:
            print('  state[%s]: %s' % (i, state))
    return chunk_size, res


# partition_space(4, 128, 4)
