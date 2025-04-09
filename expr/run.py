# %%
# 加载配置文件
from expr_utils.dataset import dataset_configs
import yaml
import os
import pickle


anay_dir = "/ws/output/analysis"
pwd = os.path.dirname(os.path.abspath(__file__))
cfg_file = os.path.join(pwd, "cfg.yml")
with open(cfg_file) as f:
    cfg = yaml.safe_load(f)

# load params
dataset = dataset_configs("/ws/output/gt")
params = []
cnt = 0
for seq in dataset["kitti"]:
    param = {**seq, **cfg}
    param["config"] = f"/tmp/cfg-{cnt}.yml"
    param["result"] = f"/ws/output/expr/{param['id']}.txt"
    param["analysis"] = f"/ws/output/analysis/{param['id']}.pkl"
    param["merge"] = 10.0
    param["idx_thd"] = 100
    param["iou_thd"] = 0.7
    param["rps"] = []
    if not os.path.exists(param["analysis"]):
        pickle.dump(param, open(param["analysis"], "wb"))
    params.append(param)
    cnt += 1

print(f"load {len(params)} configures")


# %%
# 执行算法，计算结果
import subprocess as sp

CMD = 'bash -c "source /ws/install/setup.bash && rosrun BoW3D expr -c {0} > /dev/null"'

for param in params[:4]:
    yaml.dump(param, open(param["config"], "w"))
    cmd = CMD.format(param["config"])
    print(f"run {cmd}")
    sp.run(cmd, shell=True)
    os.remove(param["config"])
