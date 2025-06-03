from datasets import load_dataset
from torch.utils.data import Dataset


class GRPODataset(Dataset):
    def __init__(
        self, dataset_id: str, split: str, extra_columns: str | None = None
    ):
        self.data = load_dataset(dataset_id, "main")[split]
        if extra_columns is None:
            extra_columns = []
        self.extra_columns = extra_columns.split(",")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        item = {"prompt": sample["prompt"]}
        if "images" in sample:
            item["images"] = sample["images"]
        for col in self.extra_columns:
            if col in sample:
                item[col] = sample[col]
        return item
