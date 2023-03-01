import cv2
import numpy as np
import os

EPSILON = 0.0000000000001
COLOR3D = "color3d"
COLORPERCH = "color per ch"


def top1Accuracy(
    querydir, supportdir,
    gridm, gridn,
    histtype, unit_r, unit_g, unit_b,
    histcache: dict,
    imgcache: dict
):
    qbasename = os.path.basename(querydir)
    sbasename = os.path.basename(supportdir)

    queryimgs = os.listdir(querydir)
    supportimgs = os.listdir(supportdir)

    numOfQueryImgs = len(queryimgs)
    numOfSupportImgs = len(supportimgs)

    correctImgs = 0
    for nameQ in queryimgs:

        imgQ = cv2.imread(os.path.join(querydir, nameQ))
        klvals = np.zeros(numOfSupportImgs, dtype=object)

        for s, nameS in enumerate(supportimgs):
            if nameS in imgcache:
                imgS = imgcache[nameS]
            else:
                imgS = cv2.imread(os.path.join(supportdir, nameS))
                imgcache[nameS] = imgS

            klvals[s] = (
                KL_from_img(
                    os.path.join(qbasename, histtype, nameQ),
                    os.path.join(sbasename, histtype, nameS),
                    imgQ, imgS,
                    gridm, gridn,
                    histcache,
                    histtype,
                    unit_r=unit_r,
                    unit_g=unit_g,
                    unit_b=unit_b,
                    maxlevel=256,
                ),
                nameS
            )

        _, nameS = np.min(klvals)
        if nameS == nameQ:
            correctImgs += 1

    return correctImgs/numOfQueryImgs


def KL_from_img(nameQ: str, nameS: str, imgQ: np.array, imgS: np.array, gridm: int, gridn: int, histcache: dict, histtype: str, **kwargs):
    histfunc = color_hist if histtype == COLOR3D else per_ch_hist
    m = imgQ.shape[0]
    n = imgQ.shape[1]
    grid: np.array = np.zeros(
        (
            int(np.ceil(m/gridm)),
            int(np.ceil(n/gridn))
        )
    )
    for i in range(0, m, gridm):
        for j in range(0, n, gridn):

            igridm = i+gridm
            jgridn = j+gridn

            keyQ = (nameQ, igridm, jgridn)
            if keyQ in histcache:
                Q = histcache[keyQ]
            else:
                Q = histfunc(
                    imgQ[i:igridm, j:jgridn],
                    kwargs["unit_r"],
                    kwargs["unit_g"],
                    kwargs["unit_b"],
                    kwargs["maxlevel"],
                )
                histcache[keyQ] = Q

            keyS = (nameS, igridm, jgridn)
            if keyS in histcache:
                S = histcache[keyS]
            else:
                S = histfunc(
                    imgS[i:igridm, j:jgridn],
                    kwargs["unit_r"],
                    kwargs["unit_g"],
                    kwargs["unit_b"],
                    kwargs["maxlevel"],
                )
                histcache[keyS] = S

            grid[i//gridm, j//gridn] = KL_div(Q, S, histtype)

    return np.average(grid)


# KL DIVERGENCE

def KL_div(Q: np.array, S: np.array, _type: str):
    Q += EPSILON
    S += EPSILON

    if _type == COLOR3D:
        return np.sum(Q*np.log10(Q/S))
    elif _type == COLORPERCH:
        p0, p1, p2 = Q/S

        return (np.sum(Q[0]*np.log10(p0)) +
                np.sum(Q[1]*np.log10(p1)) +
                np.sum(Q[2]*np.log10(p2)))/3


# CONSTRUCT HISTOGRAMS


def color_hist(img: np.array, unit_r: int, unit_g: int, unit_b: int, maxlevel: int):
    h = np.zeros(
        (
            int(np.ceil(maxlevel/unit_b)),
            int(np.ceil(maxlevel/unit_g)),
            int(np.ceil(maxlevel/unit_r))
        )
    )

    for row in img:
        for b, g, r in row:
            h[b//unit_b, g//unit_g, r//unit_r] += 1
    return h / (img.shape[0]*img.shape[1])


def per_ch_hist(img: np.array, unit_r: int, unit_g: int, unit_b: int, maxlevel: int):
    h_r = np.zeros(int(np.ceil(maxlevel/unit_r)))
    h_g = np.zeros(int(np.ceil(maxlevel/unit_g)))
    h_b = np.zeros(int(np.ceil(maxlevel/unit_b)))

    for row in img:
        for b, g, r in row:
            h_r[r//unit_r] += 1
            h_g[g//unit_g] += 1
            h_b[b//unit_b] += 1

    return np.array([h_b, h_g, h_r]) / (img.shape[0]*img.shape[1])


if __name__ == "__main__":
    histcache: dict = {}
    imgcache: dict = {}

    querydirs = [
        "dataset/query_1",
        "dataset/query_2",
        "dataset/query_3"
    ]
    support_96 = "dataset/support_96"

    queries = [os.path.basename(querydir) for querydir in querydirs]

    # 1 3D Color Histogram
    intervals = [16, 32, 64, 128]
    print("1 3D Color Histogram")
    for i, query in enumerate(queries):
        print(query)
        for interval in intervals:
            acc = top1Accuracy(
                querydirs[i], support_96,
                96, 96,
                COLOR3D, interval, interval, interval,
                histcache, imgcache
            )
            print(f"{interval}: {acc}")

    # 2 Per Channel Color Histogram
    intervals = [8, 16, 32, 64, 128]
    print("2 Per Channel Color Histogram")
    for i, query in enumerate(queries):
        print(query)
        for interval in intervals:
            acc = top1Accuracy(
                querydirs[i], support_96,
                96, 96,
                COLORPERCH, interval, interval, interval,
                histcache, imgcache
            )
            print(f"{interval}: {acc}")

    gridmns = [48, 24, 16, 12]
    print("3 Grid Based Feature Extraction - Query set 1")
    print("3d")
    for gridmn in gridmns:
        acc = top1Accuracy(
            querydirs[0], support_96,
            gridmn, gridmn,
            COLOR3D, 16, 16, 16,
            histcache, imgcache
        )
        print(f"{gridmn}: {acc}")
    print("Per Channel")
    for gridmn in gridmns:
        acc = top1Accuracy(
            querydirs[0], support_96,
            gridmn, gridmn,
            COLORPERCH, 16, 16, 16,
            histcache, imgcache
        )
        print(f"{gridmn}: {acc}")
    print("4 Grid Based Feature Extraction - Query set 2")
    print("3d")
    for gridmn in gridmns:
        acc = top1Accuracy(
            querydirs[1], support_96,
            gridmn, gridmn,
            COLOR3D, 16, 16, 16,
            histcache, imgcache
        )
        print(f"{gridmn}: {acc}")
    print("Per Channel")
    for gridmn in gridmns:
        acc = top1Accuracy(
            querydirs[1], support_96,
            gridmn, gridmn,
            COLORPERCH, 16, 16, 16,
            histcache, imgcache
        )
        print(f"{gridmn}: {acc}")
    print("5 Grid Based Feature Extraction - Query set 3")
    print("3d")
    for gridmn in gridmns:
        acc = top1Accuracy(
            querydirs[2], support_96,
            gridmn, gridmn,
            COLOR3D, 16, 16, 16,
            histcache, imgcache
        )
        print(f"{gridmn}: {acc}")
    print("Per Channel")
    for gridmn in gridmns:
        acc = top1Accuracy(
            querydirs[2], support_96,
            gridmn, gridmn,
            COLORPERCH, 16, 16, 16,
            histcache, imgcache
        )
        print(f"{gridmn}: {acc}")
