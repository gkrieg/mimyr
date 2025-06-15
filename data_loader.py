import os
import random
import scanpy as sc
import pandas as pd
import tqdm

from alignment_model import AlignementModel
from gene_exp_model import GeneExpModel

class SliceDataLoader:
    def __init__(self, mode="intra", label="subclass"):
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
        
        self.train_slices = None
        self.val_slices = None
        self.test_slices = None
        self.gene_exp_model=None
        self.density_model=None

    def load_intra_slices(self):
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
        slices1 = [sc.read_h5ad(os.path.join(input_dir, fname)) for fname in sorted_slices1]
        return slices1

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

            self.train_slices = train_slices
            self.val_slices = [val_slice]
            self.test_slices = [test_slice]

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
