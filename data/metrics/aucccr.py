from typing import List, Tuple, Callable
import random
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

scf = 1
thd = 20


def recommend_clusters(testr, d: Callable = lambda x, y: np.linalg.norm(x - y),
                       n: Callable = lambda x: 1.0 / (1.0 + (scf * x)),
                       v: Callable = lambda x: np.sqrt(scf * x),
                       p_vector: Callable = lambda x: x,
                       conv: bool = True, atom: bool = True,
                       prc: int = 20, mmt: int = 5):
    """
    :param testr:
    :param d: distance between x and y
    :param n: distance between an agent and the group'y barycenter (as measured by a distance function d)
    :param v: the value of the group of size N
    :param p_vector:
    :param conv:
    :param atom: agents are “atoms”, too small relatively to the whole system to have significant influence individually
    :param prc: the number of (random) initial values to try
    :param mmt: momentum effect
    :return: lists of clusters, each cluster contains indexes of respectable data points belonging to that cluster
    """
    ans: List[List[int]] = []
    m: int = len(testr)
    la: float = m  # affectation’s value (sum of utilities)
    k: int = 0  # number of clusters
    pmm: int = mmt
    while True:
        vla = la
        lat: float = 0
        k += 1
        for tra in range(prc):
            gen = random.Random()
            obgrp = [k for _ in range(m)]
            bv = gen.randint(0, m - 1)

            ctr = []  # [[] for _ in range(k)]
            ngrp = []  # [[] for _ in range(k)]  # new groups (in building)
            ctr.append(testr[bv])
            ngrp.append([bv])

            obgrp[bv] = 0
            ds: List[float] = np.inf * np.ones(m)

            for i in range(m):
                dv = d(testr[i], testr[bv])
                ds[i] = dv * dv

            for i in range(1, k):
                rv = gen.uniform(0.0, float(np.sum(ds)))
                nv = 0
                while ds[nv] < rv:
                    rv -= ds[nv]
                    nv += 1

                obgrp[nv] = len(ngrp)
                ctr.append(testr[nv])
                ngrp.append([nv])

                for j in range(m):
                    dist = d(testr[j], testr[nv])
                    d2 = dist * dist
                    if d2 < ds[j]:
                        ds[j] = d2

            grp = []  # groups
            # ngrp = [[] for _ in range(k)]
            vrr = 0
            while True:
                grp = ngrp.copy()
                ngrp = [[] for _ in range(k)]

                for i in range(k):
                    ctr[i] = np.zeros(len(p_vector(ctr[i])))
                    for j in range(len(grp[i])):
                        ctr[i] += p_vector(testr[grp[i][j]])
                    ctr[i] /= len(grp[i])
                if conv:
                    vr = 0.0
                    for i in range(k):
                        vg = v(len(grp[i]))
                        for p in grp[i]:
                            vr += n(d(testr[p], ctr[i])) * vg
                    if vr <= vrr:
                        break
                    vrr = vr

                nsiz = [m for _ in range(k)]  # new groups’ sizes (in building)
                bgrp = [0 for _ in range(m)]  # estimated best groups for each agent (inner loop)
                while True:
                    siz = nsiz.copy()
                    nsiz = [0 for _ in range(k)]

                    for i in range(m):
                        bintr = 1.0
                        bgrp[i] = k
                        for j in range(k):
                            intr = 0
                            if atom:
                                intr = n(d(testr[i], ctr[j])) * v(siz[j])
                            elif obgrp[i] == j:
                                intr = n(d(testr[i], ctr[j])) * v(siz[j])
                            else:
                                intr = n(d(testr[i], ((siz[j] * ctr[j]) + testr[i]) / (siz[j] + 1))) * v(siz[j] + 1)
                            if intr > bintr:
                                bgrp[i] = j
                                bintr = intr
                        if bgrp[i] < k:
                            nsiz[bgrp[i]] += 1

                    if all(x != y for x, y in zip(nsiz, siz)) and (not conv or sum(nsiz) < sum(siz)):
                        continue
                    break

                for i in range(m):
                    if bgrp[i] < k:
                        ngrp[bgrp[i]].append(i)
                obgrp = bgrp.copy()
                if all(x == y for x, y in zip(ngrp, grp)):
                    break

            lat = 0.0  # new affectation’s (just built) value (sum of utilities)
            sl = m
            for i in range(k):
                gv = v(len(grp[i]))
                for j in range(len(grp[i])):
                    lat += n(d(testr[grp[i][j]], ctr[i])) * gv
                    sl -= 1
            lat += sl
            if lat > la:
                if lat > vla:
                    ans = grp.copy()
                la = lat
        if la > vla:
            pmm = mmt
        else:
            pmm -= 1
        if pmm == 0:
            break
    ans = [an for an in ans if len(an) > 0]
    return ans


def build_vector(model, dataset):
    r = [np.zeros((0, model.layers[-1].units)) for _ in range(model.layers[-1].units)]
    for x, y in dataset:
        logits = model(x, training=False)
        for i in range(len(r)):
            r[i] = np.concatenate([r[i], logits[np.squeeze(y) == i]], axis=0)

    return np.concatenate([np.mean(ri, axis=0) if len(ri) > 0 else np.zeros(model.layers[-1].units) for ri in r])


def recommend_agent_clusters_centralized(agents, dataset, **kwargs):
    data = []
    for i in range(len(agents)):
        v = build_vector(agents[i].model, dataset)
        data.append(v)

    return recommend_clusters(data, **kwargs)


def recommend_agent_clusters_decentralized(agents, threshold=-1, **kwargs):
    data = []
    for i in range(len(agents)):
        a_data = []
        for j in range(len(agents)):
            vj = build_vector(agents[j].model, agents[i].test)
            a_data.append(vj)
        data.append(a_data)

    def euc(x, y):
        if isinstance(y, int):
            return np.linalg.norm(data[x][y] - data[x][y])
        return np.linalg.norm(data[x][x] - y)

    def projection(x):
        if isinstance(x, int):
            return data[x][x]
        return x

    if threshold > 0:
        v = lambda x: np.sqrt(threshold) if x > threshold else np.sqrt(x),
    else:
        v = lambda x: np.sqrt(x)
    return recommend_clusters(list(range(len(agents))),
                              d=euc,
                              p_vector=projection,
                              v=v,
                              **kwargs)


if __name__ == '__main__':
    # Alternative value of the group of size N (v)
    # v=lambda x: np.sqrt(scf*thd) if x > thd else np.sqrt(scf*x)
    vv = np.asarray([
        [0, 0],
        [0, 1],
        [0, 1],
        [0.1, 0]
    ])
    res = recommend_clusters(
        testr=vv,
    )
    print(res)
