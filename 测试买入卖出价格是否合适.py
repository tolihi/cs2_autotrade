import os
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================= 配置 =================
WEIGHTS = {
    "market_trend": 0.2,
    "price_ratio": 0.3,
    "price_change_1d": 0.2,
    "scratched_new_ratio": 0.1,
    "slope_min": 0.1,
    "slope_max": 0.1
}
BUY_THRESHOLD = 0.6
LOOKBACK_DAYS = 90
SLOPE_WINDOW = 30
COOLDOWN_DAYS = 7
FOCUS_CATEGORY = "枪皮"

# ================= Excel 数据读取（兼容新版列名 + 不报 KeyError） =================
def read_excel_to_goods_dict(file_paths: list) -> tuple[dict, dict]:
    """
    读取多个 Excel 文件，每个 sheet 对应一个 item，每个 item 可能包含多个 goods
    返回：
        goods_data_dict: { 'good_name': df, ... }
        goods_config: { 'good_name': {'min_expected_price': xx, 'max_expected_price': xx} }
    """

    goods_data_dict = {}
    goods_config = {}

    # 新列映射，保证统一英文列名
    col_mapping = {
        "timestamp": "timestamp",
        "price": "price",
        "volume": "volume",
        "max_price": "max_price",
        "min_price": "min_price",
        # 可选列，方便以后扩展
        "item": "item",
        "good_id": "good_id"
    }

    for file_path in file_paths:
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df_sheet = xl.parse(sheet_name)
            # 重命名列为标准列名
            df_sheet = df_sheet.rename(columns={v: k for k, v in col_mapping.items()})

            # 检查核心列
            required_cols = ["timestamp", "price", "volume", "max_price", "min_price"]
            if not all(col in df_sheet.columns for col in required_cols):
                logging.warning(f"{sheet_name} 缺少必要列 {required_cols}，跳过")
                continue

            df_sheet['timestamp'] = pd.to_datetime(df_sheet['timestamp'])

            # 分组逻辑：优先 good_id，其次 item，如果都没有就整张 sheet 作为一个饰品
            if "good_id" in df_sheet.columns:
                df_goods_groups = [(good_id, df_sheet[df_sheet["good_id"] == good_id]) for good_id in df_sheet["good_id"].unique()]
            elif "item" in df_sheet.columns:
                df_goods_groups = [(item, df_sheet[df_sheet["item"] == item]) for item in df_sheet["item"].unique()]
            else:
                df_goods_groups = [(sheet_name, df_sheet)]

            for good_name, df_good in df_goods_groups:
                df_good = df_good.reset_index(drop=True)
                goods_data_dict[good_name] = df_good

                # 自动计算最低/最高期望价
                min_price = df_good["price"].rolling(LOOKBACK_DAYS, min_periods=1).min().iloc[-1] * 0.95
                max_price = df_good["price"].rolling(LOOKBACK_DAYS, min_periods=1).max().iloc[-1] * 1.05
                goods_config[good_name] = {"min_expected_price": min_price, "max_expected_price": max_price}

    logging.info(f"已读取 {len(goods_data_dict)} 个饰品的数据")
    return goods_data_dict, goods_config






# ================= 特征生成 =================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    features = pd.DataFrame()
    features["price"] = df["price"]
    features["price_ratio"] = df["price"] / df["price"].rolling(LOOKBACK_DAYS, min_periods=1).min()
    features["price_change_1d"] = df["price"].pct_change().fillna(0)
    features["scratched_new_ratio"] = df.get("price_scratched", 1.0) / df.get("price_new", 1.0)

    slopes_min, slopes_max = [], []
    for i in range(len(df)):
        start = max(0, i - SLOPE_WINDOW)
        x = np.arange(start, i+1)
        y_min = df["price"].iloc[start:i+1].rolling(SLOPE_WINDOW, min_periods=1).min().values
        y_max = df["price"].iloc[start:i+1].rolling(SLOPE_WINDOW, min_periods=1).max().values
        slopes_min.append(np.polyfit(x, y_min, 1)[0] if len(x) > 1 else 0)
        slopes_max.append(np.polyfit(x, y_max, 1)[0] if len(x) > 1 else 0)
    features["slope_min"] = slopes_min
    features["slope_max"] = slopes_max
    features["random_noise"] = np.random.normal(0, 0.01, len(df))
    return features.fillna(method="bfill")

# ================= PyTorch 回归模型 =================
class TorchRegressor(nn.Module):
    def __init__(self, input_dim):
        super(TorchRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)   # 输入层 -> 隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)          # 隐藏层 -> 隐藏层
        self.fc3 = nn.Linear(32, 1)           # 隐藏层 -> 输出层

    def forward(self, x):
        x = self.relu(self.fc1(x))            # 第一层 + 激活函数
        x = self.relu(self.fc2(x))            # 第二层 + 激活函数
        x = self.fc3(x)                       # 输出层
        return x

def train_regression_model(X: pd.DataFrame, y: pd.Series, epochs=50, lr=0.001):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为张量
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    model = TorchRegressor(X.shape[1])
    criterion = nn.MSELoss()                  # 损失函数：均方误差
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器

    # 训练循环
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # 测试集 MSE
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mse = mean_squared_error(y_test_tensor.numpy(), y_pred.numpy())
        logging.info(f"回归模型训练完成，测试集 MSE: {mse:.4f}")
    return model

# ================= 评分计算 =================
def calculate_buy_score(features: pd.DataFrame, weights: dict) -> pd.Series:
    weighted_sum = sum(features[col] * w for col, w in weights.items() if col in features)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(weighted_sum.values.reshape(-1,1)).flatten()
    return normalized

# ================= 决策函数 =================
def evaluate_goods_regression(goods_data_dict, goods_config):
    results = []
    last_buy_time = {}

    for good_name, df in goods_data_dict.items():
        if df.empty :
            continue

        min_price = goods_config[good_name]["min_expected_price"]
        max_price = goods_config[good_name]["max_expected_price"]

        features = build_features(df)
        y = df["price"].pct_change().shift(-1).fillna(0)

        model = train_regression_model(features, y)
        with torch.no_grad():
            preds = model(torch.tensor(features.values, dtype=torch.float32)).numpy().flatten()
        features["pred"] = preds

        score = calculate_buy_score(features, WEIGHTS)

        decision = []
        for i, row in df.iterrows():
            ts = row["timestamp"]
            if good_name in last_buy_time and (ts - last_buy_time[good_name]).days < COOLDOWN_DAYS:
                decision.append("观望")
                continue
            if row["price"] < min_price:
                decision.append("买入")
                last_buy_time[good_name] = ts
            elif row["price"] > max_price:
                decision.append("卖出")
            elif score[i] > BUY_THRESHOLD:
                decision.append("买入")
                last_buy_time[good_name] = ts
            else:
                decision.append("观望")

        results.append(pd.DataFrame({
            "good_name": good_name,
            "timestamp": df["timestamp"],
            "price": df["price"],
            "score": score,
            "decision": decision
        }))

    return pd.concat(results, ignore_index=True)

# ================= 主程序 =================
def main():
    excel_files = ["gun.xlsx"]
    goods_data_dict, goods_config = read_excel_to_goods_dict(excel_files)
    results = evaluate_goods_regression(goods_data_dict, goods_config)
    output_file = "purchase_decisions.csv"
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    logging.info(f"评估结果已保存至 {output_file}")

if __name__ == "__main__":
    main()
