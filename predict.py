import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os
import hydra
from omegaconf import DictConfig
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from src.datasets import train_collate
from enum import Enum, auto
from typing import Dict, Any

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([N, 2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # データローダー設定
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    test_data = DataLoader(test_set,
                           batch_size=args.data_loader.test.batch_size,
                           shuffle=args.data_loader.test.shuffle,
                           collate_fn=collate_fn,
                           drop_last=False)

    # データセットのサイズを確認
    num_frames = len(test_set)
    print(f"Total number of frames in test set: {num_frames}")

    # モデルロード
    model = EVFlowNet(args.train).to(device)
    model_path = 'checkpoints/model_epoch_1_20240716212129.pth'  # 保存されたモデルのパス
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow = model(event_image)  # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow['flow3']), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

    # 保存したファイルの内容を確認
    loaded_flow = np.load(f"{file_name}.npy")
    print(f"Loaded optical flow shape: {loaded_flow.shape}")
    print(f"Loaded optical flow data (first 5 elements): {loaded_flow.flatten()[:5]}")

if __name__ == "__main__":
    main()
