import sys
sys.path.append("generative_transformer")

# P(region|location)
from celltype_model import CelltypeModel
from data_loader import SliceDataLoader
from biological_model import BiologicalModel2
import torch
import numpy as np
from inference import Inferernce
from evaluator import Evaluator

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import copy, os, pandas as pd, json

slice_data_loader = SliceDataLoader(mode="intra2_trial",label="subclass")
slice_data_loader.prepare()

location_model = BiologicalModel2(slice_data_loader.train_slices)
location_model.fit()


celltype_model = CelltypeModel(slice_data_loader.train_slices,slice_data_loader.gene_exp_model.num_tokens, val_slice=slice_data_loader.val_slices[0], epochs=500,learning_rate=0.001, batch_size=1024, device="cuda")

# region_model.fit()
celltype_model.load_model("best_model.pt")





OUT_CSV = "inference_results_intra2_trial.csv"

def write_row(row, path=OUT_CSV):
    """Append one row; create file with header if it doesn’t exist."""
    header = not os.path.isfile(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)

def already_done(cfg, path=OUT_CSV):
    """Skip if an identical config row is already present (based on JSON)."""
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path, usecols=cfg.keys())
    return any((df == pd.Series(cfg)).all(axis=1))

# ---- config grid ------------------------------------------------------------
base = {
    "infer_location": False,
    "location_inference_type": "model",
    "infer_subclass": True,
    "subclass_inference_type": "majority_baseline",
    "homogenize_subclass": True,
    "infer_gene_expression": True,
    "expression_inference_type": "lookup",
}

grid = [
    base,
    {**base, "infer_location": True},
    # {**base, "homogenize_subclass": True},
    # {**base, "infer_location": True, "homogenize_subclass": True},
    {**base, "expression_inference_type": "averaging"},
    {**base, "infer_location": True, "expression_inference_type": "averaging"},
    # {**base, "homogenize_subclass": True, "expression_inference_type": "averaging"},
    {**base, "infer_location": True, "expression_inference_type": "averaging"},
    # {**base, "infer_location": True, "homogenize_subclass": True, "expression_inference_type": "averaging"},
    {**base, "subclass_inference_type": "model"},
    {**base, "infer_location": True, "subclass_inference_type": "model"},
    # {**base, "homogenize_subclass": True},
    # {**base, "infer_location": True, "homogenize_subclass": True},
    {**base, "expression_inference_type": "averaging", "subclass_inference_type": "model"},
    {**base, "infer_location": True, "expression_inference_type": "averaging", "subclass_inference_type": "model"},
    # {**base, "homogenize_subclass": True, "expression_inference_type": "averaging"},
    {**base, "infer_location": True, "expression_inference_type": "averaging", "subclass_inference_type": "model"},
    # {**base, "infer_location": True, "homogenize_subclass": True, "expression_inference_type": "averaging"},

]

# ---- run / append -----------------------------------------------------------
for cfg in grid:
    if already_done(cfg):
        print("skip", cfg)
        continue

    inf = Inferernce(location_model, celltype_model, slice_data_loader, copy.deepcopy(cfg))
    pred = inf.run_inference(slice_data_loader.test_slices)
    res  = Evaluator().evaluate(pred, slice_data_loader.test_slices[0],sample=1)

    row = {**cfg, **res}              # flat dict → one CSV row
    write_row(row)
    print("wrote", cfg)