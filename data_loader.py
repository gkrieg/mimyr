import os
import random
import scanpy as sc
import pandas as pd
import tqdm
import numpy as np
from alignment_model import AlignementModel
from gene_exp_model import GeneExpModel
from generative_transformer.data_util import harmonize_dataset
import torch

class SliceDataLoader:
    def __init__(self, mode="intra", label="subclass", metadata_dir='/work/magroup/skrieger/tissue_generator/spencer_gentran/generative_transformer/metadata/'):
        """
        Args:
            mode (str): 'intra' or 'transfer'
            transfer_from (list, optional): List of slice names for training (only for transfer mode)
            transfer_to (list, optional): List of slice names for val/test (only for transfer mode)
        """
        self.mode = mode
        
        self.label = label
        
        self.input_dirs = {
            "type1": "/work/magroup/skrieger/tissue_generator/quantized_slices/subclass_z1_d338_0_rotated",
            "type2": "/work/magroup/skrieger/MERFISH_BICCN/processed_data/Zhuang-ABCA-2",
            # Add more datasets if needed
        }
        
        self.ccf_csv = "data/ccf_coordinates.csv"

        self.metadata_dir = metadata_dir
        
        self.train_slices = None
        self.val_slices = None
        self.test_slices = None
        self.gene_exp_model=None
        self.density_model=None

    def load_intra_slices(self, fast=False, fast_select=None):
        # Load only slices1
        input_dir = self.input_dirs["type1"]
        sorted_slices1 = [
            "sec_05.h5ad", "sec_06.h5ad", "sec_08.h5ad", "sec_09.h5ad", "sec_10.h5ad",
            "sec_11.h5ad", "sec_12.h5ad", "sec_13.h5ad", "sec_14.h5ad", "sec_15.h5ad",
            "sec_16.h5ad", "sec_17.h5ad", "sec_18.h5ad", "sec_19.h5ad", "sec_24.h5ad",
            "sec_25.h5ad", "sec_26.h5ad", "sec_27.h5ad", "sec_28.h5ad", "sec_29.h5ad",
            "sec_30.h5ad", "sec_31.h5ad", "sec_32.h5ad", "sec_33.h5ad", "sec_35.h5ad",
            "sec_36.h5ad", "sec_37.h5ad", "sec_38.h5ad", "sec_39.h5ad", "sec_40.h5ad",
            "sec_42.h5ad", "sec_43.h5ad", "sec_44.h5ad", "sec_45.h5ad", "sec_46.h5ad",
            "sec_47.h5ad", "sec_48.h5ad", "sec_49.h5ad", "sec_50.h5ad", "sec_51.h5ad",
            "sec_52.h5ad", "sec_54.h5ad", "sec_55.h5ad", "sec_56.h5ad", "sec_57.h5ad",
            "sec_58.h5ad", "sec_59.h5ad", "sec_60.h5ad", "sec_61.h5ad", "sec_62.h5ad",
            "sec_64.h5ad", "sec_66.h5ad", "sec_67.h5ad"
        ]
        if fast:
            slices1 = [sc.read_h5ad(os.path.join(input_dir, fname)) for i,fname in enumerate(sorted_slices1) if i in [26,27,28,29]]
        elif fast_select:
            slices1 = [sc.read_h5ad(os.path.join(input_dir, fname)) for i,fname in enumerate(sorted_slices1) if i in [fast_select-1,fast_select,fast_select+1]]
        else:
            slices1 = [sc.read_h5ad(os.path.join(input_dir, fname)) for fname in sorted_slices1]
        return slices1

    def load_zhuangn_slices(self, n=2,fast_select=None, remove_edges=True):

        print("Fast selecting", fast_select)
        
        # Load slices2
        input_dir = f"/work/magroup/skrieger/MERFISH_BICCN/processed_data/Zhuang-ABCA-{n}"
        if remove_edges:
            sorted_slices2 = sorted([
                f for f in os.listdir(input_dir) if f.endswith(".h5ad")
            ])[2:-2]
        else:
            sorted_slices2 = sorted([
                f for f in os.listdir(input_dir) if f.endswith(".h5ad")
            ])
        print(sorted_slices2)
        if fast_select:
            sorted_slices2 = [f for i,f in enumerate(sorted_slices2) if i in [fast_select-1,fast_select,fast_select+1]]
            print("fs")
        else:
            sorted_slices2 = [f for i,f in enumerate(sorted_slices2)]
        print(sorted_slices2)
        slices2 = [sc.read_h5ad(os.path.join(input_dir, fname)) for fname in tqdm.tqdm(sorted_slices2)]
        
        # CCF coordinate filtering
        df_ccf = pd.read_csv(f"/home/apdeshpa/projects/tissue-generator/data/ccf-ABCA-{n}.csv").set_index("cell_label")
        for i, slice in enumerate(slices2):
            df_filtered = df_ccf.loc[df_ccf.index.intersection(slice.obs_names)]
            df_filtered = df_filtered.reindex(slice.obs_names)
            slice.obs["x_ccf"] = df_filtered["x"]
            slice.obs["y_ccf"] = df_filtered["y"]
            slice.obs["z_ccf"] = df_filtered["z"]
            valid_mask = ~slice.obs[["x_ccf", "y_ccf", "z_ccf"]].isna().any(axis=1)
            slices2[i] = slice[valid_mask].copy()

        return slices2


    def load_diseased_slices(self):
        # Load only slices1
        input_dir = "/work/magroup/skrieger/tissue_generator/CCF_registration/ccf_aligned_Trem2_5xFAD/cleaned_versions/"
        sorted_slices1 = [
            "Trem2_5xFAD1_cleaned.h5ad", "Trem2_5xFAD2_cleaned.h5ad", "Trem2_5xFAD3_cleaned.h5ad", "Trem2_5xFAD4_cleaned.h5ad", "Trem2_5xFAD5_cleaned.h5ad"
        ]
        return [sc.read_h5ad(os.path.join(input_dir, fname)) for fname in sorted_slices1]





    def load_transfer_slices(self):
        # Load slices1
        slices1 = self.load_intra_slices()
        
        # Load slices2
        input_dir = self.input_dirs["type2"]
        sorted_slices2 = sorted([
            f for f in os.listdir(input_dir) if f.endswith(".h5ad")
        ])
        slices2 = [sc.read_h5ad(os.path.join(input_dir, fname)) for fname in tqdm.tqdm(sorted_slices2)]
        
        # CCF coordinate filtering
        df_ccf = pd.read_csv(self.ccf_csv).set_index("cell_label")
        for i, slice in enumerate(slices2):
            df_filtered = df_ccf.loc[df_ccf.index.intersection(slice.obs_names)]
            df_filtered = df_filtered.reindex(slice.obs_names)
            slice.obs["x_ccf"] = df_filtered["x"]
            slice.obs["y_ccf"] = df_filtered["y"]
            slice.obs["z_ccf"] = df_filtered["z"]
            valid_mask = ~slice.obs[["x_ccf", "y_ccf", "z_ccf"]].isna().any(axis=1)
            slices2[i] = slice[valid_mask].copy()

        return slices1, slices2

    def prepare(self):
        if self.mode == "intra":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            val_slice = slices_tokenized[5]
            test_slice = slices_tokenized[10]
            train_slices = slices_tokenized[:5] + slices_tokenized[6:10] + slices_tokenized[11:]

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None


        elif self.mode == "intra2":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            val_slice = slices_tokenized[26]
            test_slice = slices_tokenized[28]
            train_slices = slices_tokenized[:26] + slices_tokenized[27:28] + slices_tokenized[29:]
            reference_slices = [slices_tokenized[27],slices_tokenized[29]]

            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
            for s in reference_slices:
                s = harmonize_dataset(s, meta_info, edges)
            val_slice = harmonize_dataset(val_slice, meta_info, edges)
            test_slice = harmonize_dataset(test_slice, meta_info, edges)

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]
            self.reference_slices = reference_slices

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None

        elif "rq1_" in self.mode:
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            # Manual split
            test_indices = [8, 16, 29, 34, 43]
            test_indices = [test_indices[int(self.mode[-1])]]  # pick one index based on mode

            val_indices = [6, 28, 44, 14]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            # Train = everything except test + val
            train_indices = [i for i in range(len(slices_tokenized))
                            if i not in test_indices and i not in val_indices]
            train_slices = [slices_tokenized[i] for i in train_indices]

            ref_indices = []
            for i in test_indices:
                if i - 1 >= 0:
                    ref_indices.append(i - 1)
                if i + 1 < len(slices_tokenized):
                    ref_indices.append(i + 1)

            # remove duplicates, just in case
            ref_indices = sorted(set(ref_indices))
            self.reference_slices = [slices_tokenized[i] for i in ref_indices]
            

            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            new_train_slices = []
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_train_slices.append(s)
            train_slices = new_train_slices
            new_reference_slices = []
            for s in self.reference_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_reference_slices.append(s)
            reference_slices = new_reference_slices
            new_val_slices = []
            for s in val_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_val_slices.append(s)
            val_slices = new_val_slices
            new_test_slices = []
            for s in test_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_test_slices.append(s)
            test_slices = new_test_slices

            self.train_slices = train_slices
            self.val_slices = val_slices
            self.test_slices = test_slices
            self.reference_slices = reference_slices

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None

            self.gene_exp_model = GeneExpModel(train_slices + val_slices + test_slices, label=self.label)
            self.gene_exp_model.fit()
        
        

        elif self.mode == "rq1":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            test_indices = [8, 16, 29, 34, 43]
            val_indices = [6, 28, 44, 14]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            # Train = everything except test + val
            train_indices = [i for i in range(len(slices_tokenized))
                            if i not in test_indices and i not in val_indices]
            train_slices = [slices_tokenized[i] for i in train_indices]


            # references: you previously picked [27, 29], keep or adjust as needed
            # References = neighbors (-1, +1) of test slices, if valid
            ref_indices = []
            for i in test_indices:
                if i - 1 >= 0:
                    ref_indices.append(i - 1)
                if i + 1 < len(slices_tokenized):
                    ref_indices.append(i + 1)

            # remove duplicates, just in case
            ref_indices = sorted(set(ref_indices))
            self.reference_slices = [slices_tokenized[i] for i in ref_indices]

            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            new_train_slices = []
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_train_slices.append(s)
            train_slices = new_train_slices
            new_reference_slices = []
            for s in self.reference_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_reference_slices.append(s)
            reference_slices = new_reference_slices
            new_val_slices = []
            for s in val_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_val_slices.append(s)
            val_slices = new_val_slices
            new_test_slices = []
            for s in test_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_test_slices.append(s)
            test_slices = new_test_slices

            self.train_slices = train_slices
            self.val_slices = val_slices
            self.test_slices = test_slices
            self.reference_slices = reference_slices

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None



        elif self.mode == "rq2":
            # Only load slices1
            slices = self.load_zhuangn_slices(fast_select=33, n=2)

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()


            hole_seeds = [3885, 4632, 9765, 10192, 27389]

            test_slices = []
            masks = []
            self.hole_centers = []
            for hs in hole_seeds:
                print(hs)
                new_slice = slices_tokenized[1].copy()
                mask = np.linalg.norm(new_slice.obsm["aligned_spatial"]-new_slice.obsm["aligned_spatial"][hs],axis=1)<0.3
                new_slice = new_slice[mask]
                masks.append(mask)
                test_slices.append(new_slice)
                self.hole_centers.append(new_slice.obsm["aligned_spatial"].mean(0))

            #logical or all masks
            logical_or_mask = np.logical_or.reduce(masks)

            remaining_cells_slice = slices_tokenized[1][~logical_or_mask]

            #random 90 10 train val split
            train_mask = np.random.rand(len(remaining_cells_slice)) < 0.9
            train_slices = [remaining_cells_slice[train_mask]]
            val_slices = [remaining_cells_slice[~train_mask]]

            self.reference_slices = [remaining_cells_slice for t in test_slices] + [remaining_cells_slice for t in test_slices]

            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            new_train_slices = []
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_train_slices.append(s)
            train_slices = new_train_slices
            new_reference_slices = []
            for s in self.reference_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_reference_slices.append(s)
            reference_slices = new_reference_slices
            new_val_slices = []
            for s in val_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_val_slices.append(s)
            val_slices = new_val_slices
            new_test_slices = []
            for s in test_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_test_slices.append(s)
            test_slices = new_test_slices

            self.train_slices = train_slices
            self.val_slices = val_slices
            self.test_slices = test_slices
            self.reference_slices = reference_slices

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None



        elif self.mode == "rq3":
            # Only load slices1
            slices = self.load_zhuangn_slices(n=2)

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            # [15, 18,  21,  24, 29, 32, 37,  40, 42,  44]
            test_indices = [18, 24, 32, 40, 44]
            val_indices = [29]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            # Train = everything except test + val
            train_indices = [15, 21, 37, 42]
            train_slices = [slices_tokenized[i] for i in train_indices]


            # references: you previously picked [27, 29], keep or adjust as needed
            # References = neighbors (-1, +1) of test slices, if valid
            ref_indices = [15, 21, 21, 29, 29, 37,  37, 42, 42, 42]

            self.reference_slices = [slices_tokenized[i] for i in ref_indices]

            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            new_train_slices = []
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_train_slices.append(s)
            train_slices = new_train_slices
            new_reference_slices = []
            for s in self.reference_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_reference_slices.append(s)
            reference_slices = new_reference_slices
            new_val_slices = []
            for s in val_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_val_slices.append(s)
            val_slices = new_val_slices
            new_test_slices = []
            for s in test_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_test_slices.append(s)
            test_slices = new_test_slices

            self.train_slices = train_slices
            self.val_slices = val_slices
            self.test_slices = test_slices
            self.reference_slices = reference_slices

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None



        elif self.mode == "rq4":
            # Only load slices1
            slices = self.load_zhuangn_slices(n=3, remove_edges=False)

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            # [15, 18,  21,  24, 29, 32, 37,  40, 42,  44]
            test_indices = [2, 5, 11, 14, 18]
            val_indices = [7]


            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            # Train = everything except test + val
            train_indices = [0, 3, 12, 17]
            train_slices = [slices_tokenized[i] for i in train_indices]


            # references: you previously picked [27, 29], keep or adjust as needed
            # References = neighbors (-1, +1) of test slices, if valid
            ref_indices = [0, 3, 3, 7, 7, 12, 12, 17, 17, 17]

            self.reference_slices = [slices_tokenized[i] for i in ref_indices]

            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            new_train_slices = []
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_train_slices.append(s)
            train_slices = new_train_slices
            new_reference_slices = []
            for s in self.reference_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_reference_slices.append(s)
            reference_slices = new_reference_slices
            new_val_slices = []
            for s in val_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_val_slices.append(s)
            val_slices = new_val_slices
            new_test_slices = []
            for s in test_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_test_slices.append(s)
            test_slices = new_test_slices

            self.train_slices = train_slices
            self.val_slices = val_slices
            self.test_slices = test_slices
            self.reference_slices = reference_slices

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None



        elif self.mode == "rq5":
            # Only load slices1
            slices = self.load_diseased_slices()

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            # [15, 18,  21,  24, 29, 32, 37,  40, 42,  44]
            test_indices = [4]
            val_indices = [3]


            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            # Train = everything except test + val
            train_indices = [0, 1, 2]
            train_slices = [slices_tokenized[i] for i in train_indices]


            # references: you previously picked [27, 29], keep or adjust as needed
            # References = neighbors (-1, +1) of test slices, if valid
            ref_indices = [2, 3]

            self.reference_slices = [slices_tokenized[i] for i in ref_indices]

            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            new_train_slices = []
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_train_slices.append(s)
            train_slices = new_train_slices
            new_reference_slices = []
            for s in self.reference_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_reference_slices.append(s)
            reference_slices = new_reference_slices
            new_val_slices = []
            for s in val_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_val_slices.append(s)
            val_slices = new_val_slices
            new_test_slices = []
            for s in test_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_test_slices.append(s)
            test_slices = new_test_slices

            self.train_slices = train_slices
            self.val_slices = val_slices
            self.test_slices = test_slices
            self.reference_slices = reference_slices

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None

        elif self.mode == "zhuang-3":
            # Only load slices1
            slices = self.load_zhuangn_slices(n=3)

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            train_slices = slices_tokenized

            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            new_train_slices = []
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_train_slices.append(s)
            train_slices = new_train_slices

            self.train_slices = train_slices
            self.val_slices = []
            self.test_slices = []
            self.reference_slices = []

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None
        # elif "rq2" in self.mode:
        #     # Only load slices1
        #     slices = self.load_rq2_slices()
        #     # Alignment
        #     self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
        #     self.density_model.fit()
        #     aligned_slices = self.density_model.get_common_coordinate_locations()
        #     # Tokenization
        #     self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
        #     self.gene_exp_model.fit()
        #     slices_tokenized = self.gene_exp_model.get_tokenized_slices()
        #     # Manual split
        #     test_indices = [8, 16, 29, 34, 43]
        #     test_indices = [test_indices[int(self.mode[-1])]]  # pick one index based on mode
        #     val_indices = [6, 28, 44, 14]
        #     test_slices = [slices_tokenized[i] for i in test_indices]
        #     val_slices = [slices_tokenized[i] for i in val_indices]

        #     train_indices = [i for i in range(len(slices_tokenized)) if i not in test_indices and i not in val_indices]
        #     train_slices = [slices_tokenized[i] for i in train_indices]

        #     self.train_slices = train_slices
        #     self.val_slices = val_slices
        #     self.test_slices = test_slices
        #     self.reference_slices = [slices_tokenized[i] for i in ref_indices]


        elif "single_slice" in self.mode:
            # Only load slices1
            slices = self.load_zhuangn_slices(n=2, fast_select=int(self.mode[-2:]))

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            val_slice = None
            print(self.mode[-2:])
            test_slice = slices_tokenized[1]
            # train_slices = slices_tokenized[:28] + slices_tokenized[30:]

            self.train_slices = [slices_tokenized[0], slices_tokenized[2]]
            self.val_slices = [slices_tokenized[0], slices_tokenized[2]]
            self.test_slices = [test_slice]
            self.reference_slices = [slices_tokenized[0], slices_tokenized[2]]

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None


        elif self.mode == "intra2_trial":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            val_slice = slices_tokenized[28]
            test_slice = slices_tokenized[29]
            train_slices = slices_tokenized[:28] + slices_tokenized[30:]

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]
            self.reference_slices = [slices_tokenized[27],slices_tokenized[28],slices_tokenized[30],slices_tokenized[31]]

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None


        elif self.mode == "intra2_trial_hole":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(
                slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True
            )
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split: hold out 1000-cell central hole from slice 28
            val_slice = slices_tokenized[28]
            slice28 = slices_tokenized[29]

            # --- Create a central hole of 1000 cells based on 3D center ---
            coords = slice28.obsm["aligned_spatial"]  # shape (n_cells, 3)
            center = coords.mean(axis=0)
            dists = ((coords - center) ** 2).sum(axis=1)
            sorted_idxs = dists.argsort()

            hole_idxs = sorted_idxs[:1000]
            keep_idxs = sorted_idxs[1000:]

            slice28_hole = slice28[hole_idxs].copy()
            slice28_rest = slice28[keep_idxs].copy()

            # Construct splits
            train_slices = (
                slices_tokenized[:28] + slices_tokenized[30:] + [slice28_rest]
            )

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [slice28_hole]
            self.reference_slices = [slices_tokenized[27],slices_tokenized[28],slices_tokenized[30],slices_tokenized[31]]

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None

        elif self.mode == "intra3":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            val_slice = slices_tokenized[26]
            test_slice = slices_tokenized[27]
            train_slices = slices_tokenized[:26] + slices_tokenized[28:]

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]
            self.reference_slices = [slices_tokenized[27],slices_tokenized[29]]

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None


        elif self.mode == "intra2_hole":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(
                slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True
            )
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split: hold out 1000-cell central hole from slice 28
            val_slice = slices_tokenized[26]
            slice28 = slices_tokenized[28]

            # --- Create a central hole of 1000 cells based on 3D center ---
            coords = slice28.obsm["aligned_spatial"]  # shape (n_cells, 3)
            center = coords.mean(axis=0)
            dists = ((coords - center) ** 2).sum(axis=1)
            sorted_idxs = dists.argsort()

            hole_idxs = sorted_idxs[:1000]
            keep_idxs = sorted_idxs[1000:]

            slice28_hole = slice28[hole_idxs].copy()
            slice28_rest = slice28[keep_idxs].copy()

            # Construct splits
            train_slices = (
                slices_tokenized[:26] + slices_tokenized[27:28] + slices_tokenized[29:] + [slice28_rest]
            )

            reference_slices = [slices_tokenized[27],slices_tokenized[29]]

            
            meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            new_train_slices = []
            for s in train_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_train_slices.append(s)
            train_slices = new_train_slices
            new_reference_slices = []
            for s in reference_slices:
                s = harmonize_dataset(s, meta_info, edges)
                new_reference_slices.append(s)
            reference_slices = new_reference_slices
            val_slice = harmonize_dataset(val_slice, meta_info, edges)
            test_slice_hole = harmonize_dataset(slice28_hole, meta_info, edges)

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice_hole]
            self.reference_slices = reference_slices

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None

            self.gene_exp_model = GeneExpModel(train_slices, label=self.label)
            self.gene_exp_model.fit()

        elif self.mode == "intra3_hole":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(
                slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True
            )
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split: hold out 1000-cell central hole from slice 28
            val_slice = slices_tokenized[26]
            slice27 = slices_tokenized[27]

            # --- Create a central hole of 1000 cells based on 3D center ---
            coords = slice27.obsm["aligned_spatial"]  # shape (n_cells, 3)
            center = coords.mean(axis=0)
            dists = ((coords - center) ** 2).sum(axis=1)
            sorted_idxs = dists.argsort()

            hole_idxs = sorted_idxs[:1000]
            keep_idxs = sorted_idxs[1000:]

            slice27_hole = slice27[hole_idxs].copy()
            slice27_rest = slice27[keep_idxs].copy()

            # Construct splits
            train_slices = (
                slices_tokenized[:26] + slices_tokenized[28:] + [slice27_rest]
            )

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [slice27_hole]
            self.reference_slices = [slices_tokenized[27],slices_tokenized[29]]

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None
            
        elif self.mode == "zhilei":
            # Only load slices1
            slices = self.load_intra_slices()

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            val_slice = slices_tokenized[26]
            test_slice = slices_tokenized[28]
            train_slices = [slices_tokenized[27],slices_tokenized[29]]
            
            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None

        elif self.mode == "debug":
            # Only load slices1
            slices = self.load_intra_slices(fast=True)

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            val_slice = slices_tokenized[0]
            test_slice = slices_tokenized[1]
            train_slices = [slices_tokenized[2],slices_tokenized[3]]
            self.reference_slices = [slices_tokenized[2],slices_tokenized[3]]

            # meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            # edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            # new_train_slices = []
            # for s in train_slices:
            #     s = harmonize_dataset(s, meta_info, edges)
            #     new_train_slices.append(s)
            # train_slices = new_train_slices
            # new_reference_slices = []
            # for s in self.reference_slices:
            #     s = harmonize_dataset(s, meta_info, edges)
            #     new_reference_slices.append(s)
            # reference_slices = new_reference_slices
            # new_val_slices = []
            # for s in val_slices:
            #     s = harmonize_dataset(s, meta_info, edges)
            #     new_val_slices.append(s)
            # val_slices = new_val_slices
            # new_test_slices = []
            # for s in test_slices:
            #     s = harmonize_dataset(s, meta_info, edges)
            #     new_test_slices.append(s)
            # test_slices = new_test_slices 

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]

            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None


        elif self.mode == "superdebug":
            # Only load slices1
            slices = self.load_intra_slices(fast=True)

            # Alignment
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()

            # Manual split
            val_slice = slices_tokenized[0]
            test_slice = slices_tokenized[1]
            train_slices = [slices_tokenized[2],slices_tokenized[3]]
            self.reference_slices = [slices_tokenized[2],slices_tokenized[3]]

            # meta_info = torch.load(f'{self.metadata_dir}4hierarchy_metainfo_mouse_geneunion2_DAG.pt')
            # edges = [f'{self.metadata_dir}edges_x.pkl',f'{self.metadata_dir}edges_y.pkl',f'{self.metadata_dir}edges_z.pkl']
            # new_train_slices = []
            # for s in train_slices:
            #     s = harmonize_dataset(s, meta_info, edges)
            #     new_train_slices.append(s)
            # train_slices = new_train_slices
            # new_reference_slices = []
            # for s in self.reference_slices:
            #     s = harmonize_dataset(s, meta_info, edges)
            #     new_reference_slices.append(s)
            # reference_slices = new_reference_slices
            # new_val_slices = []
            # for s in val_slices:
            #     s = harmonize_dataset(s, meta_info, edges)
            #     new_val_slices.append(s)
            # val_slices = new_val_slices
            # new_test_slices = []
            # for s in test_slices:
            #     s = harmonize_dataset(s, meta_info, edges)
            #     new_test_slices.append(s)
            # test_slices = new_test_slices 

            self.train_slices = train_slices
            self.val_slices = [val_slice]

            #subsample 1000 random cells from test slice
            test_slice = test_slice[np.random.choice(test_slice.n_obs, 1000, replace=False)]            
            self.test_slices = [test_slice]


            # No fine-tuning slices in intra mode
            self.fine_tune_train_slices = None
            self.fine_tune_val_slices = None
            self.fine_tune_test_slices = None




        elif self.mode == "transfer":
            # Load both slices1 and slices2
            slices1, slices2 = self.load_transfer_slices()

            # Alignment (on both)
            slices = slices1 + slices2
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()
            # self.gene_exp_model = GeneExpModel(aligned_slices[len(slices1):], use_subclass=True)            ### WANT THE TOKENIZER TO FINALLY BE ACCURATE
            # self.gene_exp_model.fit()

            # slices_tokenized[:len(slices1)] are slices1
            # slices_tokenized[len(slices1):] are slices2
            slices1_tokenized = slices_tokenized[:len(slices1)]
            slices2_tokenized = slices_tokenized[len(slices1):]

            # === Step 1: Build train/val/test from slices1 ===
            val_slice = slices1_tokenized[5]
            test_slice = slices1_tokenized[10]
            train_slices = slices1_tokenized[:5] + slices1_tokenized[6:10] + slices1_tokenized[11:]

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]

            # === Step 2: Build fine-tune train/val/test from slices2 ===
            # Hardcode selection based on your request

            fine_tune_train_indices = [7, 32]  # 7 and 32
            fine_tune_val_index = 29           # 29

            all_indices = list(range(len(slices2_tokenized)))

            # Build the sets
            fine_tune_train = [slices2_tokenized[i] for i in fine_tune_train_indices]
            fine_tune_val = [slices2_tokenized[fine_tune_val_index]]

            fine_tune_test_indices = [i for i in all_indices if i not in fine_tune_train_indices and i != fine_tune_val_index]
            fine_tune_test = [slices2_tokenized[i] for i in fine_tune_test_indices]

            self.fine_tune_train_slices = fine_tune_train
            self.fine_tune_val_slices = fine_tune_val
            self.fine_tune_test_slices = fine_tune_test


        elif self.mode == "transfer2":
            # Load both slices1 and slices2
            slices1, slices2 = self.load_transfer_slices()

            # Alignment (on both)
            slices = slices1 + slices2
            self.density_model = AlignementModel(slices, z_posn=[-1, 0, 1], pin_key="parcellation_structure", use_ccf=True)
            self.density_model.fit()
            aligned_slices = self.density_model.get_common_coordinate_locations()

            # Tokenization
            self.gene_exp_model = GeneExpModel(aligned_slices, label=self.label)
            self.gene_exp_model.fit()
            slices_tokenized = self.gene_exp_model.get_tokenized_slices()
            # self.gene_exp_model = GeneExpModel(aligned_slices[len(slices1):], use_subclass=True)            ### WANT THE TOKENIZER TO FINALLY BE ACCURATE
            # self.gene_exp_model.fit()

            # slices_tokenized[:len(slices1)] are slices1
            # slices_tokenized[len(slices1):] are slices2
            slices1_tokenized = slices_tokenized[:len(slices1)]
            slices2_tokenized = slices_tokenized[len(slices1):]

            # === Step 1: Build train/val/test from slices1 ===
            val_slice = slices1_tokenized[5]
            test_slice = slices1_tokenized[10]
            train_slices = slices1_tokenized[:5] + slices1_tokenized[6:10] + slices1_tokenized[11:]

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]

            # === Step 2: Build fine-tune train/val/test from slices2 ===
            # Hardcode selection based on your request

            fine_tune_train_indices = [31, 34]  # 7 and 32
            fine_tune_val_index = 33           # 29


            all_indices = list(range(len(slices2_tokenized)))

            # Build the sets
            fine_tune_train = [slices2_tokenized[i] for i in fine_tune_train_indices]
            fine_tune_val = [slices2_tokenized[fine_tune_val_index]]

            fine_tune_test_indices = [i for i in all_indices if i not in fine_tune_train_indices and i != fine_tune_val_index]
            fine_tune_test = [slices2_tokenized[32]]

            self.fine_tune_train_slices = fine_tune_train
            self.fine_tune_val_slices = fine_tune_val
            self.fine_tune_test_slices = fine_tune_test

            slicesz=[]
            for slice in self.train_slices:
                slicesz.append((slice.obsm["aligned_spatial"][:,2].mean()-self.fine_tune_test_slices[0].obsm["aligned_spatial"][:,2].mean())**2)
            self.reference_slices=[self.train_slices[i] for i in np.argsort(slicesz)[:2]]


        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Print summary
        print(f"Prepared data:")
        print(f" - Train: {len(self.train_slices)} slices")
        print(f" - Val: {len(self.val_slices)} slices")
        print(f" - Test: {len(self.test_slices)} slices")
        if self.fine_tune_train_slices is not None:
            print(f" - Fine-tune Train: {len(self.fine_tune_train_slices)} slices")
            print(f" - Fine-tune Val: {len(self.fine_tune_val_slices)} slices")
            print(f" - Fine-tune Test: {len(self.fine_tune_test_slices)} slices")