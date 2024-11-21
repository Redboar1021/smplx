import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

class FlattenedPoseDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []

        for file_path in file_paths:
            data = np.load(file_path)
            poses = torch.tensor(data['poses'], dtype=torch.float32)
            trans = torch.tensor(data['trans'], dtype=torch.float32)
            combined = torch.cat([poses, trans], dim=1)  # (165 + 3 = 168)
            self.data.append(combined)
        
        self.data = torch.cat(self.data, dim=0)  # フラット化して1つのテンソルに結合

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# データセットのパスを指定
file_paths = [
    "/root/development/time-machine/dataset/TotalCapture/s1/acting1_stageii.npz",
    "/root/development/time-machine/dataset/TotalCapture/s1/acting2_stageii.npz",
    "/root/development/time-machine/dataset/TotalCapture/s1/acting3_stageii.npz",
    "/root/development/time-machine/dataset/TotalCapture/s1/freestyle1_stageii.npz",
    "/root/development/time-machine/dataset/TotalCapture/s1/freestyle2_stageii.npz",
    "/root/development/time-machine/dataset/TotalCapture/s1/freestyle3_stageii.npz"
]

# データセット作成
dataset = FlattenedPoseDataset(file_paths)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# データ確認
print(f"Dataset size: {len(dataset)}")
print(f"Sample data shape: {dataset[0].shape}")

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)  # 平均
        self.fc_logvar = nn.Linear(64, latent_dim)  # 分散の対数
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
        
# 損失関数
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)  # 再構成損失
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KLダイバージェンス
    return recon_loss + kld_loss

# モデル学習関数
def train_vae(dataset, epochs=50, batch_size=64, latent_dim=16, learning_rate=0.001, save_path="vae_model.pth"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = dataset[0].shape[0]  # 入力次元（168）
    model = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")
    
    # モデル保存
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

def load_model(input_dim, latent_dim, path="vae_model.pth"):
    model = VAE(input_dim, latent_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
# モデルをロード
#loaded_model = load_model(input_dim=168, latent_dim=16, path="vae_model.pth")
# サンプリングと生成
""" def generate_samples(model, latent_dim, num_samples=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)  # 標準正規分布からサンプリング
        generated_data = model.decode(z)  # デコーダで生成
    return generated_data """

# 生成データの取得
#generated_data = generate_samples(loaded_model, latent_dim=16, num_samples=5)
print("Generated Data:")
#print(generated_data)

vae_model = train_vae(dataset, epochs=10, latent_dim=16, save_path="vae_model.pth")

# 165次元のポーズパラメータの設定
num_frames = 1  # フレーム数
pose_params = torch.zeros((num_frames, 165), device='cpu')  # ここに実際のポーズデータを設定
"""  """
# 各ポーズパラメータの分割
# ボディと手のジョイント数を定義
NUM_BODY_JOINTS = 21  # 例: 21個のボディジョイント
NUM_HAND_JOINTS = 15  # 例: 15個の手のジョイント

# サンプルのポーズパラメータ (バッチサイズ1, 165次元)
pose_params = torch.randn(1, 165)

# グローバルオリエンテーション
global_orient = pose_params[:, :3]
global_orient = global_orient.reshape(1, 3, 3)

# ボディポーズ
body_pose = pose_params[:, 3:3 + NUM_BODY_JOINTS * 3]
body_pose = body_pose.reshape(-1, NUM_BODY_JOINTS, 3, 3)

# 左手ポーズ
left_hand_pose = pose_params[:, 3 + NUM_BODY_JOINTS * 3:3 + NUM_BODY_JOINTS * 3 + NUM_HAND_JOINTS * 3]
left_hand_pose = left_hand_pose.reshape(-1, NUM_HAND_JOINTS, 3, 3)

# 右手ポーズ
right_hand_pose = pose_params[:, 3 + NUM_BODY_JOINTS * 3 + NUM_HAND_JOINTS * 3:3 + NUM_BODY_JOINTS * 3 + 2 * NUM_HAND_JOINTS * 3]
right_hand_pose = right_hand_pose.reshape(-1, NUM_HAND_JOINTS, 3, 3)

# 顎のポーズ
jaw_pose = pose_params[:, -9:-6]
jaw_pose = jaw_pose.reshape(-1, 1, 3, 3)

# 左目のポーズ
leye_pose = pose_params[:, -6:-3]
leye_pose = leye_pose.reshape(-1, 1, 3, 3)

# 右目のポーズ
reye_pose = pose_params[:, -3:]
reye_pose = reye_pose.reshape(-1, 1, 3, 3)

from smplx.body_models import SMPLX
import torch

# モデルファイルのパスを指定
model_path = '/root/development/time-machine/models/smplx'

# SMPLXクラスのインスタンスを作成
smplx_model = SMPLX(model_path=model_path)

output = smplx_model.forward(global_orient=global_orient, body_pose=body_pose, left_hand_pose=left_hand_pose,
               right_hand_pose=right_hand_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose)
print(output.vertices)

# 必要な入力テンソルを作成
# 例: ポーズ、ベータ、トランスレーションなど
# pose = torch.zeros([1, smplx_model.NUM_JOINTS * 3])  # 例としてゼロテンソルを使用
# betas = torch.zeros([1, 10])  # 例としてゼロテンソルを使用
# transl = torch.zeros([1, 3])  # 例としてゼロテンソルを使用

# # forwardメソッドを呼び出して出力を取得
# output = smplx_model.forward(pose=pose, betas=betas, transl=transl)

# # 出力を表示
# print(output)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np





# output.verticesのデータを取得
vertices = output.vertices.detach().cpu().numpy()
faces = smplx_model.faces_tensor.cpu().numpy()

# 3Dプロットの作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# x, y, z座標を取得
x = vertices[:, :, 0]
x = x.reshape(-1)
y = vertices[:, :, 1]
y = y.reshape(-1)
z = vertices[:, :, 2]
z = z.reshape(-1)

# # 3D散布図をプロット
# ax.scatter(x, y, z, c='r', marker='o')
# メッシュをプロット
ax.plot_trisurf(x, y, z, triangles=faces, cmap='viridis', edgecolor='none')

# ラベルの設定
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 各軸の範囲を設定して縮尺を統一
max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

mid_x = (x.max() + x.min()) * 0.5
mid_y = (y.max() + y.min()) * 0.5
mid_z = (z.max() + z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# プロットの表示
plt.show()