# %%
# 加载配置文件
from expr_utils.dataset import dataset_configs
import yaml
import os


anay_dir = "/ws/output/analysis"
pwd = os.path.dirname(os.path.abspath(__file__))
cfg_file = os.path.join(pwd, "cfg.yml")
with open(cfg_file) as f:
    cfg = yaml.safe_load(f)

# load params
dataset = dataset_configs("/ws/output/gt")
params = []
cnt = 0
for _, seqs in dataset.items():
    for seq in seqs:
        param = {**seq, **cfg}
        param["config"] = f"/tmp/cfg-{cnt}.yml"
        param["output"] = f"/ws/output/expr/{param['id']}.txt"
        params.append(param)
        cnt += 1

print(f"load {len(params)} configures")


# %%
# 执行算法，计算结果
import subprocess as sp

CMD = 'bash -c "source /ws/install/setup.bash && rosrun BoW3D expr -c {0} > /dev/null"'

for param in params[1:4]:
    yaml.dump(param, open(param["config"], "w"))
    cmd = CMD.format(param["config"])
    print(f"run {cmd}")
    sp.run(cmd, shell=True)
    os.remove(param["config"])

# %% 计算 PR
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pickle


def gtnum(tps: np.ndarray, gts: np.ndarray, dis_thd: float) -> np.ndarray:
    """
    Recalculate the number of ground truth pairs to reasonably evaluate algorithm performance.

    The ground truth is calculated based on 3D coordinates, resulting in numerous matches near loop closure points.
    Feature-based methods using point cloud features may lose some matches, which is acceptable in practice.
    This function treats the ID pairs of matching results as 3D coordinates and considers all nearby ground truth ID pairs as a single ground truth result.
    This approach reduces the unreasonable number of ground truth pairs and provides a more rational basis for performance evaluation.

    Parameters
    ----------
    tps : np.ndarray [n, 2]
        The matching ID pairs of the algorithm.
    gts : np.ndarray [m, 2]
        The ground truth ID pairs.
    dis_thd : float
        The threshold of distance for region merging.

    Returns
    -------
    np.ndarray [k, 2]
        The recalculated ground truth ID pairs.
    """
    ret = []
    seen = set()
    tp_points = np.hstack((tps, np.zeros((len(tps), 1))))
    tree = o3d.geometry.KDTreeFlann(tp_points.T)
    for gt in gts:
        p = np.array([gt[0], gt[1], 0.0])
        [_, idx, _] = tree.search_knn_vector_3d(p, 1)
        id = idx[0]
        err = tp_points[id] - p
        dis = abs(err[0]) + abs(err[1])
        # far from tp or not seen will be counted as valid gt
        if dis > dis_thd or id not in seen:
            ret.append([int(p.item(0)), int(p.item(1))])
        # close to tp and not seen will be added to seen
        if dis <= dis_thd and id not in seen:
            seen.add(id)
    return np.array(ret)


id = 3
params[id]["IDX_THD"] = 1800
params[id]["IOU_THD"] = 0.5
params[id]["MERGE_RNG"] = 70.0
IOU_THD = params[id]["IOU_THD"]
IDX_THD = params[id]["IDX_THD"]
MERGE_RNG = params[id]["MERGE_RNG"]
res = np.loadtxt(params[id]["output"], delimiter=",", comments="#")
gts = np.loadtxt(params[id]["ground_truth"], delimiter=",", comments="#")
gts = gts[np.logical_and(gts[:, 0] - gts[:, 1] > IDX_THD, gts[:, 2] > IOU_THD)]
print(len(gts))
max_score = np.max(res[:, 2])
rps = []
for score in np.linspace(max_score, 1e-3, 100):
    filtered_res = res[res[:, 2] > score]
    if len(filtered_res) == 0:
        continue
    tp = np.sum(filtered_res[:, 3] > IOU_THD).item(0)
    filtered_gts = gtnum(filtered_res[:, :2], gts, MERGE_RNG)
    gt_num = max(len(filtered_gts), tp)
    recall = tp / gt_num
    precision = tp / len(filtered_res)
    if len(rps) == 0 or recall >= rps[-1][0]:
        rps.append((recall, precision, tp, gt_num, len(filtered_res)))

plt.plot([rp[0] for rp in rps], [rp[1] for rp in rps], marker="o")
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.grid(True)
plt.title(params[id]["id"])

pickle.dump((params[id], rps), open(f"{anay_dir}/{id:02d}.pkl", "wb"))


# %%
# 加载数据
import pickle
import os
from pprint import pprint

rpss = {}
for root, dirs, files in os.walk(anay_dir):
    files.sort()
    for file in files:
        cfg, rps = pickle.load(open(os.path.join(anay_dir, file), "rb"))
        rpss[cfg["id"]] = rps
print(f"load {len(rpss)} results")

# %%
# 画图
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
axes = axes.flatten()

for ax, name in zip(axes, rpss):
    print(name)
    rps = np.array(rpss[name])
    ax.plot(rps[:, 0], rps[:, 1], marker="o")
    ax.set_title(name)
    ax.grid(True)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)


fig.text(0.5, 0.04, "Recall", ha="center")
fig.text(0.04, 0.5, "Precision", va="center", rotation="vertical")

plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
plt.show()
