"""The MIMYR Python API: generate, train, save, and evaluate on AnnData sections.

    import anndata as ad
    import mimyr

    model = mimyr.Mimyr.from_pretrained("merscope")
    refs = [ad.read_h5ad("sec_13.h5ad"), ad.read_h5ad("sec_15.h5ad")]
    pred = model.generate(reference=refs, position=10.9)        # x_ccf of the new section
    scores = mimyr.evaluate(pred, ad.read_h5ad("sec_14.h5ad"))  # if you have the real one

A section is an AnnData with one row per cell: Allen CCF coordinates in
obs["x_ccf"], obs["y_ccf"], obs["z_ccf"] (mm), an Allen Brain Cell Atlas cluster
label in obs["cluster"], and expression in X. `validate` checks this.
"""

import os
import json
import random
import re
import shutil
import pickle as pkl

import numpy as np
import pandas as pd
import anndata as ad
import torch

from mimyr import pretrained
from mimyr._resources import METADATA_DIR, META_INFO_FILE, META_INFO_PATH, ID_TO_SUBCLASS_PATH
from mimyr.data_modes import ORIENTATIONS, section_axis

COORDS = ("x_ccf", "y_ccf", "z_ccf")
# aligned_spatial (the models' coordinate frame) stacks the CCF axes as [z, y, x].
AXIS_COLUMN = {"z_ccf": 0, "y_ccf": 1, "x_ccf": 2}
HIERARCHY = ("class", "subclass", "supertype", "cluster")
# Default values for section metadata the models are prompted with.
DEFAULT_METADATA = {"technology": "M550", "species": "mouse", "disease_state": "healthy"}
DEFAULT_METRICS = ("soft_accuracy", "soft_spearman_correlation", "soft_f1")

# Inference settings and their defaults (main.py's). Presets override some of them,
# and generate() accepts any of them as keyword arguments.
INFERENCE_DEFAULTS = {
    "expression_weak_teacher_forcing": False,
    "wtf_strength_gene": 0.0,
    "wtf_strength_bin": 0.0,
    "wtf_strength_value": 0.0,
    "wtf_alpha_gene": 0.0,
    "wtf_alpha_bin": 0.0,
    "wtf_bin_sigma": 0.0,
    "wtf_unexpressed_mode": "none",
    "top_k": 5,
    "expression_select_max_index_margin": 0.01,
    "cluster_temperature": 1.0,
    "cluster_prior_alpha": 0.0,
}
SAVE_FORMAT = 1
MANIFEST = "mimyr.json"


# ----------------------------------------------------------------- data
def _as_list(sections):
    return [sections] if isinstance(sections, ad.AnnData) else list(sections)


def _is_prepared(section):
    return "mimyr" in section.uns and section.uns["mimyr"].get("prepared", False)


def _vocabulary():
    meta_info = torch.load(META_INFO_PATH)
    with open(ID_TO_SUBCLASS_PATH, "rb") as f:
        labels = set(pkl.load(f).values())
    return meta_info, labels


def validate(sections, label="cluster", raise_on_error=True):
    """Check that sections have what MIMYR reads, and say what is wrong if not.

    Returns a list of (level, message) with level "error" or "warning". Errors
    stop `prepare`; warnings describe data MIMYR will use but may handle badly.
    With raise_on_error, raises ValueError listing the errors.
    """
    meta_info, known_labels = _vocabulary()
    panel = set(meta_info["gene_set"])
    problems = []
    for i, s in enumerate(_as_list(sections)):
        name = f"section {i}"

        def report(level, message):
            problems.append((level, f"{name}: {message}"))

        if _is_prepared(s):
            continue
        missing = [c for c in COORDS if c not in s.obs.columns]
        if missing:
            report("error", f"obs is missing {', '.join(missing)} (Allen CCF coordinates in mm)")
        else:
            xyz = s.obs[list(COORDS)].to_numpy(dtype=float)
            n_nan = int(np.isnan(xyz).any(axis=1).sum())
            if n_nan:
                report("error", f"{n_nan} cells have no CCF coordinates; drop them first")
            elif np.nanmax(xyz) > 20:
                report("error", f"CCF coordinates go up to {np.nanmax(xyz):.0f}; MIMYR expects mm "
                       "(the mouse brain spans about 0-13), so these look like micrometres")
        if label not in s.obs.columns:
            report("error", f"obs has no {label!r} column of cell-type labels")
        else:
            labels = s.obs[label].astype(str)
            unknown = ~labels.isin(known_labels)
            if unknown.all():
                report("error", f"none of the {label!r} labels are Allen Brain Cell Atlas clusters "
                       "(e.g. '0001 CLA-EPd-CTX Car3 Glut_1'), the vocabulary the models use")
            elif unknown.any():
                report("warning", f"{int(unknown.sum())} of {s.n_obs} cells have {label!r} labels "
                       "outside the model vocabulary; they are relabelled as the most common type "
                       f"(e.g. {labels[unknown].iloc[0]!r})")
            if label != "cluster":
                gaps = [h for h in HIERARCHY if h not in s.obs.columns]
                if gaps:
                    report("error", f"obs is missing {', '.join(gaps)}; with label={label!r} the "
                           "hierarchy cannot be filled in from the cluster")
        overlap = len(panel.intersection(s.var_names))
        if overlap == 0:
            report("error", f"no genes in common with the model's {len(panel)}-gene panel; "
                   "var_names must be gene symbols")
        elif overlap < 50:
            report("warning", f"only {overlap} genes in common with the model's {len(panel)}-gene "
                   "panel; reference guidance and evaluation use only these")
        for column, values in (("technology", meta_info["technology"]),
                               ("species", meta_info["species"]),
                               ("disease_state", meta_info["disease_state"])):
            if column in s.obs.columns:
                bad = sorted(set(s.obs[column].astype(str)) - set(values))
                if bad:
                    report("warning", f"obs[{column!r}] values {bad} are not in the model vocabulary "
                           f"({', '.join(values)}) and are ignored")
    if raise_on_error:
        errors = [m for level, m in problems if level == "error"]
        if errors:
            raise ValueError("invalid sections:\n  " + "\n  ".join(errors))
    return problems


def _fill_hierarchy(obs):
    """Fill in missing class / subclass / supertype from obs["cluster"]."""
    gaps = [h for h in HIERARCHY[:3] if h not in obs.columns]
    if gaps:
        with open(os.path.join(METADATA_DIR, "hierarchy.pkl"), "rb") as f:
            hierarchy = pkl.load(f)
        levels = obs["cluster"].astype(str).map(lambda v: hierarchy.get(("cluster", v)))
        for h in gaps:
            j = HIERARCHY.index(h)
            obs[h] = levels.map(lambda t: t[j] if t else None)


def _slice_loader(orientation="coronal", label="cluster", omit_x=False):
    from mimyr.data_loader import SliceDataLoader

    return SliceDataLoader(
        mode=orientation, label=label, cfg={"meta_info": META_INFO_FILE},
        metadata_dir=METADATA_DIR, omit_x=omit_x,
    )


def prepare(sections, label="cluster", omit_x=False, technology=None):
    """Return copies of `sections` in the form the models read.

    This is the processing main.py applies to the paper's data: coordinates
    stacked into obsm["aligned_spatial"], cell types tokenized into
    obs["token"], coordinates binned, expression log1p-transformed if it looks
    like raw counts, cells with fewer than 10 genes dropped, and genes padded to
    the model's panel. Missing class/subclass/supertype are filled in from the
    cluster label, and missing technology / species / disease_state from
    `technology` (default "M550") and "mouse" / "healthy".

    Sections that are already prepared are returned as they are. `generate`,
    `train`, and `evaluate` call this themselves; call it once yourself to
    reuse prepared sections across calls.
    """
    single = isinstance(sections, ad.AnnData)
    sections = _as_list(sections)
    todo = [i for i, s in enumerate(sections) if not _is_prepared(s)]
    if todo:
        validate([sections[i] for i in todo], label=label)
        copies = []
        for i in todo:
            s = sections[i]
            # New obs/var; X is shared, and the processing below replaces it rather
            # than modifying it, so the caller's sections are left untouched.
            c = ad.AnnData(X=s.X, obs=s.obs.copy(), var=s.var.copy())
            if label == "cluster":
                _fill_hierarchy(c.obs)
            for column, default in DEFAULT_METADATA.items():
                if column not in c.obs.columns:
                    c.obs[column] = technology or default if column == "technology" else default
            copies.append(c)
        sdl = _slice_loader(label=label, omit_x=omit_x)
        done = sdl._harmonize_slice_lists([], [], [], sdl._align_and_tokenize_slices(copies))[3]
        sections = list(sections)
        for i, s in zip(todo, done):
            s.uns["mimyr"] = {"prepared": True, "label": label, "omit_x": bool(omit_x)}
            sections[i] = s
    return sections[0] if single else sections


# ----------------------------------------------------------------- model
def _epoch_of(path):
    m = re.search(r"epoch(\d+)", os.path.basename(path or ""))
    return int(m.group(1)) if m else None


def _seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Mimyr:
    """A MIMYR model: location diffusion model, cell-type classifier, and
    expression transformer, plus the inference settings to use them with.

    Create one with `Mimyr.from_pretrained` (a paper model, or a folder written
    by `save`) or directly from three checkpoint paths.
    """

    def __init__(self, location_checkpoint, celltype_checkpoint, expression_checkpoint,
                 orientation="coronal", settings=None, omit_x=False, name=None):
        if orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}, not {orientation!r}")
        self.location_checkpoint = location_checkpoint
        self.celltype_checkpoint = celltype_checkpoint
        self.expression_checkpoint = expression_checkpoint
        self.orientation = orientation
        self.settings = {**INFERENCE_DEFAULTS, **(settings or {})}
        self.omit_x = bool(omit_x)
        self.name = name
        self._models = None

    def __repr__(self):
        return (f"Mimyr({self.name or 'custom'}, orientation={self.orientation!r}, "
                f"expression={os.path.basename(self.expression_checkpoint)!r})")

    # ------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, name_or_path=pretrained.DEFAULT_PRESET):
        """Load a paper model by name (downloaded from the Hugging Face Hub and
        cached) or a model folder written by `save`.

        Names: merscope (Fig. 2, the default), zhuang1, zhuang2-finetuned
        (Fig. 3), zhuang3-finetuned (Fig. 4, sagittal). Each comes with the
        inference settings used for the paper's numbers.
        """
        if os.path.isdir(name_or_path):
            return cls.load(name_or_path)
        if name_or_path not in pretrained.PRESETS:
            raise ValueError(f"unknown model {name_or_path!r}: not a folder, and not one of "
                             f"{', '.join(pretrained.PRESETS)}")
        preset = pretrained.PRESETS[name_or_path]
        settings = dict(preset["settings"])
        omit_x = settings.pop("omit_x", False)
        settings.pop("metric_filter_by_gt", None)  # an evaluation setting; evaluate() defaults to it
        return cls(
            *(pretrained.checkpoint(name_or_path, role) for role in ("location", "celltype", "expression")),
            orientation=preset["orientation"], settings=settings, omit_x=omit_x, name=name_or_path,
        )

    def save(self, path):
        """Write the three checkpoints and the settings to the folder `path`."""
        os.makedirs(path, exist_ok=True)
        epoch = _epoch_of(self.expression_checkpoint)
        files = {
            "location": "location.pt",
            "celltype": "celltype.pt",
            # Training resumes its epoch count from the number in this name.
            "expression": f"expression_epoch{epoch}.pt" if epoch else "expression.pt",
        }
        for role, filename in files.items():
            src = getattr(self, f"{role}_checkpoint")
            dst = os.path.join(path, filename)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copyfile(src, dst)
        manifest = {
            "format": SAVE_FORMAT, "name": self.name, "orientation": self.orientation,
            "omit_x": self.omit_x, "settings": self.settings, "files": files,
        }
        with open(os.path.join(path, MANIFEST), "w") as f:
            json.dump(manifest, f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        """Load a model folder written by `save`."""
        with open(os.path.join(path, MANIFEST)) as f:
            manifest = json.load(f)
        if manifest.get("format") != SAVE_FORMAT:
            raise ValueError(f"{path} was saved in format {manifest.get('format')}; this version "
                             f"of mimyr reads format {SAVE_FORMAT}")
        files = manifest["files"]
        return cls(
            *(os.path.join(path, files[role]) for role in ("location", "celltype", "expression")),
            orientation=manifest["orientation"], settings=manifest["settings"],
            omit_x=manifest["omit_x"], name=manifest.get("name"),
        )

    def _combined(self):
        if self._models is None:
            from mimyr.models.combined_model import CombinedModel

            self._models = CombinedModel(
                location_model_checkpoint=self.location_checkpoint,
                celltype_model_checkpoint=self.celltype_checkpoint,
            )
        return self._models

    # ------------------------------------------------------------- generation
    def generate(self, reference=None, position=None, n_cells=None, *, bounds=None,
                 reference_guidance=None, location="model", cell_types="model",
                 expression="model", seed=None, **options):
        """Generate a tissue section.

        reference: the measured sections to condition on, normally the ones on
            either side of the missing section. Used for reference guidance, the
            default extent and size of the section, and the baselines.
        position: where to put the section, as a coordinate in mm along the
            section axis (x_ccf for coronal models, z_ccf for sagittal), or a
            held-out section to regenerate (its position, extent, and number of
            cells are used; its expression is not).
        n_cells: number of cells (default: the held-out section's, else the mean
            of the references').
        bounds: the in-plane extent, as {"y_ccf": (lo, hi), "z_ccf": (lo, hi)}
            for a coronal section ({"y_ccf": ..., "x_ccf": ...} for sagittal).
            Default: the extent of the held-out section, else of the references.
        reference_guidance: strength in [0, 100] with which expression is pulled
            toward the nearest same-type reference cell; 0 turns it off. Default:
            the model's setting (75 for merscope).
        location / cell_types / expression: "model", or a baseline for that
            stage: location "closest_slice" (copy the nearest reference's cell
            positions), cell_types "majority_baseline" (majority of the 20
            nearest reference cells), expression "lookup" (copy the nearest
            same-type reference cell).
        seed: seeds Python, NumPy, and torch first, for a reproducible section.
        options: other inference settings (see INFERENCE_DEFAULTS).

        Returns an AnnData with the generated cells: positions in
        obs["x_ccf"/"y_ccf"/"z_ccf"] and obsm["spatial"] (ordered z, y, x), cell
        types in obs["cluster"] (and class/subclass/supertype, obs["token"]),
        and expression over the model's gene panel in X.
        """
        from mimyr.inference import Inference, SectionTarget, SECTION_METADATA

        unknown = set(options) - set(INFERENCE_DEFAULTS)
        if unknown:
            raise TypeError(f"unknown inference settings {sorted(unknown)}; "
                            f"choose from {', '.join(INFERENCE_DEFAULTS)}")
        if position is None:
            raise TypeError("generate() needs a position (a coordinate, or a held-out section)")
        config = {**self.settings, **options}
        if reference_guidance is not None:
            config["expression_weak_teacher_forcing"] = reference_guidance > 0
            for part in ("gene", "bin", "value"):
                config[f"wtf_strength_{part}"] = float(reference_guidance)
        needs_reference = (config["expression_weak_teacher_forcing"] or location != "model"
                           or cell_types != "model" or expression != "model")

        refs = prepare(reference, omit_x=self.omit_x) if reference is not None else []
        if needs_reference and not refs:
            raise ValueError("this needs reference sections: reference guidance is on or a "
                             "baseline was chosen (pass reference_guidance=0 to go without)")
        pool = ad.concat(refs) if refs else None
        ax = section_axis(self.orientation)

        if isinstance(position, ad.AnnData):
            held_out = prepare(position, omit_x=self.omit_x)
            target = SectionTarget.from_section(held_out)
            target.ground_truth = None  # regenerate it, never read it
            target.bin_edge_source = pool
            if n_cells is not None:
                target.obs_names = pd.Index([f"cell_{i}" for i in range(n_cells)])
                target.metadata = {k: v.mode().iloc[0] for k, v in target.metadata.items()}
        else:
            if pool is None and (bounds is None or n_cells is None):
                raise ValueError("without reference sections, pass both bounds and n_cells")
            centre = np.zeros(3)
            centre[ax] = float(position)
            if pool is not None:
                xyz = pool.obsm["aligned_spatial"]
                lower, upper = xyz.min(0), xyz.max(0)
            else:
                lower, upper = np.zeros(3), np.zeros(3)
            for coord, (lo, hi) in (bounds or {}).items():
                if coord not in AXIS_COLUMN or AXIS_COLUMN[coord] == ax % 3:
                    raise ValueError(f"bounds keys are the two in-plane axes of a "
                                     f"{self.orientation} section, not {coord!r}")
                lower[AXIS_COLUMN[coord]], upper[AXIS_COLUMN[coord]] = lo, hi
            if n_cells is None:
                n_cells = int(round(np.mean([r.n_obs for r in refs])))
            metadata = {}
            if pool is not None:
                metadata = {k: pool.obs[k].mode().iloc[0] for k in SECTION_METADATA if k in pool.obs.columns}
            target = SectionTarget(
                centre=centre, lower=lower, upper=upper,
                obs_names=pd.Index([f"cell_{i}" for i in range(n_cells)]),
                metadata=metadata,
                var_names=pool.var_names if pool is not None else pd.Index([]),
                bin_edge_source=pool,
            )

        config.update(
            data_mode=self.orientation,
            location_inference_type=location,
            cluster_inference_type=cell_types,
            expression_inference_type=expression,
            expression_model_checkpoint=self.expression_checkpoint,
            meta_info=META_INFO_FILE,
            omit_x=self.omit_x,
            full_gene_panel=True,
            guidance_signal=0.0,
            artifact_dir=None,
        )
        sdl = _slice_loader(self.orientation, omit_x=self.omit_x)
        sdl.reference_slices, sdl.test_slices = refs, []
        if pool is not None:
            sdl._ref_pool_cache = (refs, pool)
        sdl.gene_exp_model = _label_map()

        _seed(seed)
        pred = Inference(self._combined(), None, sdl, config).generate(target)

        if "spatial" in pred.obsm:
            for coord, col in AXIS_COLUMN.items():
                pred.obs[coord] = np.asarray(pred.obsm["spatial"])[:, col]
        # The expression model names the cell types itself; the baselines leave tokens.
        if "token" in pred.obs.columns and "cluster" not in pred.obs.columns:
            pred.obs["cluster"] = sdl.gene_exp_model.get_label_from_token(pred.obs["token"].values)
            _fill_hierarchy(pred.obs)
        pred.uns["mimyr"] = {"orientation": self.orientation, "model": self.name or "custom",
                             "reference_guidance": config["wtf_strength_gene"]
                             if config["expression_weak_teacher_forcing"] else 0.0}
        return pred

    # ------------------------------------------------------------- training
    def train(self, train, val, output_dir, *, components=("celltype", "expression"),
              epochs=10, batch_size=8, lr=5e-5, celltype_epochs=1000, save_every=None,
              location_epochs=400, location_lr=2e-4, **expression_options):
        """Fine-tune the model on your sections, starting from its current weights.

        train, val: sections to train and validate on (validation sections set
            early stopping for the cell-type classifier and are scored each epoch
            for the expression model).
        output_dir: where checkpoints go; the model then points at the new ones,
            so call `save` afterwards to keep it as a folder.
        components: any of "location", "celltype" and "expression". The
            pretrained location model already covers the whole mouse brain in CCF
            space, so "location" is opt-in; it trains coronal models only.
        epochs, batch_size, lr: for the expression model; celltype_epochs caps
            the classifier's training (it stops early on validation);
            location_epochs and location_lr are for the location model, which
            has no validation step and keeps a checkpoint every 20 epochs.
        save_every: also keep the expression checkpoint every this many epochs.
        expression_options: other expression-trainer settings, by their
            argument names in finetune_mimyr (e.g. xyz_noise=True, lambda_val=1.0).
            num_workers=4 loads batches in worker processes, which start by
            re-importing your script: put its code under
            `if __name__ == "__main__":` first, or it runs again in each worker.

        Run the script with torchrun to train the expression model on several
        GPUs. Weights & Biases logging is off unless WANDB_MODE is set.
        """
        components = (components,) if isinstance(components, str) else tuple(components)
        bad = set(components) - {"location", "celltype", "expression"}
        if bad:
            raise ValueError(f"unknown components {sorted(bad)}; "
                             "choose from location, celltype and expression")
        if "location" in components and self.orientation != "coronal":
            raise NotImplementedError(f"location training supports coronal models only, "
                                      f"and this one is {self.orientation}")
        import torch.distributed as dist

        distributed = dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1
        if distributed and not dist.is_initialized():
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        rank = dist.get_rank() if distributed else 0

        os.makedirs(output_dir, exist_ok=True)
        train = _as_list(prepare(train, omit_x=self.omit_x))
        val = _as_list(prepare(val, omit_x=self.omit_x))

        if "location" in components:
            path = os.path.join(output_dir, "location.pt")
            if rank == 0:
                self._train_location(ad.concat(train), output_dir, path,
                                     epochs=location_epochs, lr=location_lr)
            if distributed:
                dist.barrier()
            self.location_checkpoint = path
            self._models = None

        if "celltype" in components:
            path = os.path.join(output_dir, "celltype.pt")
            if rank == 0:
                celltype = self._combined().celltype_model
                celltype.fit(ad.concat(train), val_adata=ad.concat(val), epochs=celltype_epochs,
                             batch_size=1024, save_path=path)
                if celltype.best_model is None:
                    celltype.save_model(path)
            if distributed:
                dist.barrier()
            self.celltype_checkpoint = path
            self._models = None

        if "expression" in components:
            self.expression_checkpoint = self._train_expression(
                train, val, os.path.join(output_dir, "expression"), rank, distributed,
                epochs=epochs, batch_size=batch_size, lr=lr, save_every=save_every,
                **expression_options,
            )
        return self

    def _train_location(self, sections, output_dir, path, *, epochs, lr):
        from mimyr.models.combined_model import TrainConfig
        from mimyr.models.diffusion_model import DDPMTrainer

        # aligned_spatial is [z_ccf, y_ccf, x_ccf]. The model learns the coronal
        # plane (z, y) conditioned on [4, 5, x_ccf, 0, 0, 1], which is how
        # infer_location conditions it for a coronal section.
        xyz = np.asarray(sections.obsm["aligned_spatial"], dtype=np.float32)
        cond = np.zeros((len(xyz), 6), dtype=np.float32)
        cond[:, 0], cond[:, 1], cond[:, 2], cond[:, 5] = 4.0, 5.0, xyz[:, 2], 1.0

        # Normalization: DDPMTrainer ignores these sections' statistics and
        # normalizes coordinates and conditions with constants hardcoded from the
        # whole-mouse-brain CCF training set. Inference un-normalizes with the
        # same constants (CombinedModel reads only the weights, not the stats
        # saved below), so train and sample agree. That suits CCF-registered
        # mouse data in mm; data in another frame or scale would need new
        # constants, changed in DDPMTrainer for both training and sampling.
        ckpt_dir = os.path.join(output_dir, "location")
        os.makedirs(ckpt_dir, exist_ok=True)
        cfg = TrainConfig(epochs=epochs, batch_size=4096, lr=lr, ckpt_dir=ckpt_dir,
                          ckpt_prefix="ddpm_2d_checkpoint", viz=False)
        trainer = DDPMTrainer(xyz, cond, cfg)
        if self.location_checkpoint is not None:  # fine-tune from the current weights
            current = self._combined().trainer
            trainer.model.load_state_dict(current.model.state_dict())
            trainer.ema.shadow = {k: v.detach().clone() for k, v in current.ema.shadow.items()}
        trainer.train()
        torch.save({
            "model": trainer.model.state_dict(), "ema": trainer.ema.shadow, "cfg": cfg,
            "coord_mean": trainer.mean, "coord_std": trainer.std,
            "cond_mean": trainer.cond_mean, "cond_std": trainer.cond_std,
        }, path)

    def _train_expression(self, train, val, out, rank, distributed, *, epochs, batch_size, lr,
                          save_every, **options):
        from mimyr.models.generative_transformer import (
            get_expression_parser, train_expression_model,
        )

        # The trainer reads preprocessed sections from train.* / val.* in its output folder.
        if any(os.path.exists(os.path.join(out, f"{s}.slaf")) for s in ("train", "val")):
            raise FileExistsError(f"{out} holds train.slaf / val.slaf from an earlier run, which "
                                  "the trainer would read instead of these sections; use another output_dir")
        if rank == 0:
            os.makedirs(out, exist_ok=True)
            ad.concat(train, join="outer").write_h5ad(os.path.join(out, "train.h5ad"))
            ad.concat(val, join="outer").write_h5ad(os.path.join(out, "val.h5ad"))
        if distributed:
            import torch.distributed as dist
            dist.barrier()

        start = (_epoch_of(self.expression_checkpoint) or 0) + 1
        last = start + epochs - 1
        args = get_expression_parser().parse_args([
            "--meta-info", META_INFO_PATH, "--data-mode", "api", "--data-dir", out,
            # trailing separator: the trainer writes f"{output_dir}bin_edges.pt"
            "--output-dir", os.path.join(out, ""),
        ])
        unknown = set(options) - set(vars(args))
        if unknown:
            raise TypeError(f"unknown expression training settings {sorted(unknown)}")
        vars(args).update(
            ckp_path=self.expression_checkpoint, from_finetuned=True, epochs=last,
            batch_size=batch_size, lr=lr, save_frequency=save_every or last,
            device="cuda" if torch.cuda.is_available() else "cpu",
            # main.py's settings: no per-cell sampling weights file, fixed seed
            disable_sampling_probs=True, seed=42, omit_x=self.omit_x,
            # in-process loading, so an unguarded script is not re-run by workers
            num_workers=0,
        )
        vars(args).update(options)
        os.environ.setdefault("WANDB_MODE", "disabled")
        train_expression_model(args)
        return os.path.join(out, f"epoch{last}_model.pt")


def _label_map():
    """Token <-> Allen cluster name map, as SliceDataLoader builds it."""
    from mimyr.gene_exp_model import GeneExpModel

    labels = GeneExpModel([], label="cluster")
    with open(ID_TO_SUBCLASS_PATH, "rb") as f:
        labels.id_to_subclass = pkl.load(f)
    labels.subclass_to_id = {v: k for k, v in labels.id_to_subclass.items()}
    return labels


# ----------------------------------------------------------------- evaluation
def evaluate(pred, ground_truth, metrics=DEFAULT_METRICS, sample=100, filter_by_gt=True,
             gene_set=None, orientation=None):
    """Score a generated section against the real one.

    metrics: any of mimyr.evaluator.METRICS (default soft accuracy, soft
        Spearman correlation, soft F1, each at several neighbourhood radii).
    sample: percentage of real cells the metrics are computed on.
    filter_by_gt: score every gene expressed in the real section (the paper's
        setting); False keeps only genes expressed in both.
    gene_set: restrict expression metrics to these genes.
    orientation: "coronal" or "sagittal" (default: what generate() recorded).

    Returns {metric column: value}, e.g. "soft_f1_radius@0.05".
    """
    from mimyr.evaluator import Evaluator, METRICS

    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; choose from {', '.join(METRICS)}")
    orientation = orientation or pred.uns.get("mimyr", {}).get("orientation", "coronal")
    gt = prepare(ground_truth)
    config = {"metrics": metrics, "data_mode": orientation, "meta_info": META_INFO_FILE}
    evaluator = Evaluator(config)
    if gene_set is not None:
        evaluator.gene_set = list(gene_set)
    # evaluate() reduces the coordinate arrays in place, so give it copies
    res = evaluator.evaluate(pred.copy(), gt.copy(), sample=sample, filter_by_gt=filter_by_gt)
    return {k: float(v) for k, v in res.items()}
