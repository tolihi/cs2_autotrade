import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

"""
WEIGHTS 注释：详细说明每个参数的含义
支持 sheet 中多个 goods：df_sheet.groupby("good_name") 将每个饰品单独处理
每个饰品独立期望价：动态计算最低/最高价格
冷却期、回归、权重评分、最终决策：保持原有逻辑
多 Excel 文件支持：可在 excel_files 列表中添加多个文件
"""

# ================= 配置 =================
WEIGHTS = {
    "market_trend": 0.2,  # 当前市场总体趋势，正值表示涨、负值表示跌
    "price_ratio": 0.3,  # 当前价格 / 过去 LOOKBACK_DAYS 最低价，用于判断溢价
    "price_change_1d": 0.2,  # 最近 1 天涨跌幅
    "scratched_new_ratio": 0.1,  # 略磨价格 / 崭新价格，用于判断品相影响
    "slope_min": 0.1,  # 历史最低价斜率，判断长期上涨趋势
    "slope_max": 0.1  # 历史最高价斜率，判断长期上涨趋势
}

BUY_THRESHOLD = 0.6  # 权重评分高于该值建议购买
LOOKBACK_DAYS = 90  # 历史回溯天数，用于计算价格溢价
SLOPE_WINDOW = 30  # 斜率计算窗口
COOLDOWN_DAYS = 7  # 买入冷却期
FOCUS_CATEGORY = "枪皮"  # 只关注该类别饰品


# ================= Excel 数据读取 =================
def read_excel_to_goods_dict(file_paths: list) -> tuple[dict, dict]:
    """
    读取多个 Excel 文件，每个 sheet 对应一个 item，每个 item 可能包含多个 goods
    返回：
        goods_data_dict: { '饰品名': df, ... }
        goods_config: { '饰品名': {'min_expected_price': xx, 'max_expected_price': xx} }
    """
    goods_data_dict = {}
    goods_config = {}

    for file_path in file_paths:
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df_sheet = xl.parse(sheet_name)
            df_sheet['timestamp'] = pd.to_datetime(df_sheet['timestamp'])

            # 假设 df_sheet 包含列: ['timestamp', 'good_name', 'price', 'category', ...]
            for good_name, df_good in df_sheet.groupby("good_name"):
                df_good = df_good.reset_index(drop=True)
                goods_data_dict[good_name] = df_good

                # 自动计算最低/最高期望价（可调系数）
                min_price = df_good["price"].rolling(LOOKBACK_DAYS, min_periods=1).min().iloc[-1] * 0.95
                max_price = df_good["price"].rolling(LOOKBACK_DAYS, min_periods=1).max().iloc[-1] * 1.05
                goods_config[good_name] = {"min_expected_price": min_price, "max_expected_price": max_price}

    logging.info(f"已读取 {len(goods_data_dict)} 个饰品的数据")
    return goods_data_dict, goods_config


# ================= 特征生成 =================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个饰品构建回归特征
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    features = pd.DataFrame()

    features["price"] = df["price"]  # 当前价格

    # 价格溢价
    features["price_ratio"] = df["price"] / df["price"].rolling(LOOKBACK_DAYS, min_periods=1).min()

    # 当前涨跌百分比
    features["price_change_1d"] = df["price"].pct_change().fillna(0)

    # 品相倍数（略磨 / 崭新）
    if "price_scratched" in df.columns and "price_new" in df.columns:
        features["scratched_new_ratio"] = df["price_scratched"] / df["price_new"]
    else:
        features["scratched_new_ratio"] = 1.0

    # 历史价格斜率
    slopes_min, slopes_max = [], []
    for i in range(len(df)):
        start = max(0, i - SLOPE_WINDOW)
        x = np.arange(start, i + 1)
        y_min = df["price"].iloc[start:i + 1].rolling(SLOPE_WINDOW, min_periods=1).min().values
        y_max = df["price"].iloc[start:i + 1].rolling(SLOPE_WINDOW, min_periods=1).max().values
        slopes_min.append(np.polyfit(x, y_min, 1)[0] if len(x) > 1 else 0)
        slopes_max.append(np.polyfit(x, y_max, 1)[0] if len(x) > 1 else 0)
    features["slope_min"] = slopes_min
    features["slope_max"] = slopes_max

    # 随机扰动
    features["random_noise"] = np.random.normal(0, 0.01, len(df))

    return features.fillna(method="bfill")


# ================= 模型训练 =================
def train_regression_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """
    训练回归模型预测未来涨幅
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info(f"回归模型训练完成，测试集 MSE: {mean_squared_error(y_test, y_pred):.4f}")
    return model


# ================= 评分计算 =================
def calculate_buy_score(features: pd.DataFrame, weights: dict) -> pd.Series:
    """
    对各个特征加权并归一化，生成购买分数
    """
    weighted_sum = sum(features[col] * w for col, w in weights.items() if col in features)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(weighted_sum.values.reshape(-1, 1)).flatten()
    return normalized


# ================= 决策函数 =================
def evaluate_goods_regression(goods_data_dict, goods_config):
    """
    对每个饰品进行回归预测，并生成买入/卖出/观望决策
    """
    results = []
    last_buy_time = {}  # 冷却期控制

    for good_name, df in goods_data_dict.items():
        if df.empty or df.get("category", [None])[0] != FOCUS_CATEGORY:
            continue

        min_price = goods_config[good_name]["min_expected_price"]
        max_price = goods_config[good_name]["max_expected_price"]

        features = build_features(df)
        y = df["price"].pct_change().shift(-1).fillna(0)
        model = train_regression_model(features, y)
        pred_future = model.predict(features)
        features["pred"] = pred_future
        score = calculate_buy_score(features, WEIGHTS)

        decision = []
        for i, row in df.iterrows():
            ts = row["timestamp"]

            # 冷却期判断
            if good_name in last_buy_time and (ts - last_buy_time[good_name]).days < COOLDOWN_DAYS:
                decision.append("观望")
                continue

            # 强制买入/卖出规则
            if row["price"] < min_price:
                decision.append("买入")
                last_buy_time[good_name] = ts
            elif row["price"] > max_price:
                decision.append("卖出")
            # 权重评分判断
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
    # Excel 文件列表（可扩展为多个）
    excel_files = ["all_goods_wide.xlsx"]
    goods_data_dict, goods_config = read_excel_to_goods_dict(excel_files)

    results = evaluate_goods_regression(goods_data_dict, goods_config)

    # 保存最终决策
    output_file = "purchase_decisions.csv"
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    logging.info(f"评估结果已保存至 {output_file}")


if __name__ == "__main__":
    main()
