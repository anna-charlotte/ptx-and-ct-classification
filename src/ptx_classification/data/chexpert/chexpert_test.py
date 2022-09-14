from ptx_classification.data.chexpert.chexpert import CheXpertDataModule, CheXpertDataset
from ptx_classification.utils import get_data_dir

data_dir = get_data_dir()


class TestCheXpertDataModule:
    def test_get_splits(self) -> None:
        dataset = CheXpertDataset(
            root=data_dir,
            version="original",
            class_labels=["Pneumothorax"],
            frontal_lateral="Frontal",
            ap_pa="all",
        )
        datamodule = CheXpertDataModule(
            dataset=dataset,
            batch_size=1,
            train_val_test_split=(20, 15, 11),
        )


class TestCheXpertDataset:
    def test_load_without_cache(self) -> None:
        chexpert = CheXpertDataset(
            root=data_dir,
            version="original",
            class_labels=["Pneumothorax"],
            frontal_lateral="Frontal",
            ap_pa="all",
        )
        assert len(chexpert.train_image_paths) == 45
