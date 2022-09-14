from ptx_classification.data.explore_data import plot_venn_diagram
from ptx_classification.data.ptx_cohorts.ptx_cohorts import PtxCohortsCiipDataset
from ptx_classification.utils import get_data_dir


class TestPtxCohortsDataset:
    def test_load(self):
        data_dir = get_data_dir()
        dataset = PtxCohortsCiipDataset(root=data_dir)
        df_labels = dataset.get_labels()
        df_ptx_pos = df_labels[df_labels["Pneumothorax"] == 1.0]
        print(f"len(df_ptx_pos) = {len(df_ptx_pos)}")

        df_ct_pos = df_labels[df_labels["Chest Tube"] == 1.0]
        print(f"len(df_ct_pos) = {len(df_ct_pos)}")

        all_image_ids = df_labels["image_id"].tolist()
        print(f"len(all_image_ids) = {len(all_image_ids)}")

        ptx_pos_ct_pos = [
            img_id
            for img_id in all_image_ids
            if img_id in df_ptx_pos["image_id"].tolist()
            and img_id in df_ct_pos["image_id"].tolist()
        ]
        print(f"len(ptx_pos_ct_pos) = {len(ptx_pos_ct_pos)}")

        plot_venn_diagram(
            df_labels[["image_id", "Pneumothorax", "Chest Tube"]],
        )
