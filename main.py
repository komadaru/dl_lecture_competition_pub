import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
import torch.nn.functional as F
import pickle

print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Version: ", torch.version.cuda)
print("cuDNN Version: ", torch.backends.cudnn.version())


# print("Torchaudio Version: ", torchaudio.__version__)

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def compute_epe_error(pred_flows: Dict[str, torch.Tensor], gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    pred_flow = pred_flows['flow3']
    epe = torch.norm(pred_flow - gt_flow, p=2, dim=1).mean()
    return epe


def compute_multiscale_loss(pred_flows: Dict[str, torch.Tensor], target_flow: torch.Tensor):
    total_loss = 0
    for scale, pred_flow in pred_flows.items():
        # Scale factorの逆数でターゲットフローをリサイズ

        # target_flowの形状を確認し、必要に応じて変換
        if target_flow.dim() == 5:  # 5次元テンソルの場合 (N, T, C, H, W)
            target_flow = target_flow.view(-1, *target_flow.shape[2:])  # 4次元に変換 (N*T, C, H, W)

        # pred_flowの形状を確認し、必要に応じて変換
        if pred_flow.dim() == 5:  # 5次元テンソルの場合
            pred_flow = pred_flow.view(-1, *pred_flow.shape[2:])  # 4次元に変換

        scaled_target = F.interpolate(target_flow, size=pred_flow.shape[-2:], mode='bilinear', align_corners=True)
        scaled_target = scaled_target.to(pred_flow.dtype)  # Ensure the same dtype

        # 各スケールのMSE損失を計算
        scale_loss = F.mse_loss(pred_flow, scaled_target)

        # デバッグのために各スケールの損失を表示
        print(f"Scale {scale}, Loss: {scale_loss.item()}")

        total_loss += scale_loss

    # スケール数で正規化
    total_loss /= len(pred_flows)

    return total_loss


def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())


# 前処理を行い、前処理済みデータを保存する関数
def preprocess_and_save(loader, save_dir):
    preprocessed_train_dataset = loader.get_preprocessed_train_dataset()
    preprocessed_test_dataset = loader.get_preprocessed_test_dataset()

    # データをリストに変換して保存
    train_data_list = [sample for sample in DataLoader(preprocessed_train_dataset, batch_size=1)]
    test_data_list = [sample for sample in DataLoader(preprocessed_test_dataset, batch_size=1)]

    with open(save_dir / 'preprocessed_train_dataset.pkl', 'wb') as f:
        pickle.dump(train_data_list, f)

    with open(save_dir / 'preprocessed_test_dataset.pkl', 'wb') as f:
        pickle.dump(test_data_list, f)

# 前処理済みデータを読み込む関数
def load_preprocessed_data(save_dir):
    with open(save_dir / 'preprocessed_train_dataset.pkl', 'rb') as f:
        preprocessed_train_dataset = pickle.load(f)

    with open(save_dir / 'preprocessed_test_dataset.pkl', 'rb') as f:
        preprocessed_test_dataset = pickle.load(f)

    return preprocessed_train_dataset, preprocessed_test_dataset

# 前処理済みデータを再構築するためのカスタムデータセットクラス
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''

    # ------------------
    #    Dataloader
    # ------------------
    dataset_path = Path(args.dataset_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # DatasetProviderを作成
    loader = DatasetProvider(
        dataset_path=dataset_path,
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4,
    )

    # データの前処理と保存
    preprocess_and_save(loader, save_dir)

    # 前処理済みデータを読み込み
    preprocessed_train_dataset, preprocessed_test_dataset = load_preprocessed_data(save_dir)

    collate_fn = train_collate
    train_data = DataLoader(preprocessed_train_dataset, batch_size=args.data_loader.train.batch_size,
                            shuffle=args.data_loader.train.shuffle, collate_fn=collate_fn)
    test_data = DataLoader(preprocessed_test_dataset, batch_size=args.data_loader.test.batch_size,
                           shuffle=args.data_loader.test.shuffle, collate_fn=collate_fn)

    print(f"length of train: {len(train_data)}")
    print(f"length of test: {len(test_data)}")

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない

    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate,
                                 weight_decay=args.train.weight_decay)
    # ------------------
    #   Start training
    # ------------------
    model.train()
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch + 1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)  # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
            # flow = model(event_image) # [B, 2, 480, 640]
            flow_dict = model(event_image)  # 各スケールのフローを含む辞書
            loss: torch.Tensor = compute_multiscale_loss(flow_dict, ground_truth_flow)
            epe_error = compute_epe_error(flow_dict, ground_truth_flow).item()
            print(f"batch {i} loss: {epe_error}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


            # Check if the average loss is below the threshold
            if epe_error < 4.0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.train.low_lr
            elif epe_error < 3.5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.train.low_lr2
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}')

        # Save the model at the end of each epoch
        current_time = time.strftime("%Y%m%d%H%M%S")
        model_path = f"checkpoints/model_epoch_{epoch + 1}_{current_time}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            # batch_flow = model(event_image) # [1, 2, 480, 640]
            batch_flow_dict = model(event_image)
            batch_flow = batch_flow_dict['flow3']  # 最後のスケールのフロー
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)


if __name__ == "__main__":
    main()
