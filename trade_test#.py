import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LOOKBACK_DAYS = 90
SLOPE_WINDOW = 30
COOLDOWN_DAYS = 7
BUY_PROB_THRESHOLD = 0.6
FOCUS_CATEGORY = "枪皮"
MIN_PARA = 1.05
MAX_PARA = 0.95
EMA_WINDOW = 14

# 初始成本权重
COST_MATRIX = {
    "FN": 2.0,
    "FP": 1.0
}

# ================== 数据读取 ==================
def read_excel_to_goods_dict(file_paths: list) -> tuple[dict, dict]:
    goods_data_dict = {}
    goods_config = {}
    for file_path in file_paths:
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            if "timestamp" not in df.columns or "price" not in df.columns:
                logging.warning(f"{sheet_name} 缺少必要列，跳过")
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values("timestamp").reset_index(drop=True)

            goods_data_dict[sheet_name] = df
            min_price = df["price"].rolling(LOOKBACK_DAYS, min_periods=1).min().iloc[-1] * MIN_PARA
            max_price = df["price"].rolling(LOOKBACK_DAYS, min_periods=1).max().iloc[-1] * MAX_PARA
            goods_config[sheet_name] = {"min_expected_price": min_price, "max_expected_price": max_price}
    return goods_data_dict, goods_config

# ================== 特征构建 ==================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    features = pd.DataFrame()
    features["price"] = df["price"]
    features["price_lag1"] = df["price"].shift(1).bfill()
    features["price_lag3"] = df["price"].shift(3).bfill()
    features["price_lag7"] = df["price"].shift(7).bfill()
    features["price_ratio"] = df["price"] / df["price"].rolling(LOOKBACK_DAYS, min_periods=1).min()
    features["price_change_1d"] = df["price"].pct_change().fillna(0)
    features["scratched_new_ratio"] = df.get("price_scratched", 1.0) / df.get("price_new", 1.0)

    # EMA、Bollinger、Momentum
    features["ema_14"] = df["price"].ewm(span=EMA_WINDOW, adjust=False).mean()
    features["boll_upper"] = df["price"].rolling(20).mean() + 2*df["price"].rolling(20).std()
    features["boll_lower"] = df["price"].rolling(20).mean() - 2*df["price"].rolling(20).std()
    features["momentum_7"] = df["price"] - df["price"].shift(7)

    slopes = []
    for i in range(len(df)):
        start = max(0, i - SLOPE_WINDOW)
        x = np.arange(start, i + 1)
        y = df["price"].iloc[start:i + 1].values
        slopes.append(np.polyfit(x, y, 1)[0] if len(x) > 1 else 0)
    features["trend_slope"] = slopes
    features["ma_7"] = df["price"].rolling(7, min_periods=1).mean()
    features["volatility_7"] = df["price"].rolling(7, min_periods=1).std().fillna(0)
    features["random_noise"] = np.random.normal(0, 0.01, len(df))
    return features.ffill().bfill()

# ================== 模型 ==================
class BuySellClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.sigmoid(self.out(x))

def train_classifier(X, y, epochs=300, lr=0.001, fn_weight=2.0, fp_weight=1.0):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)
    model = BuySellClassifier(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        weight = torch.where(y_tensor==1, fn_weight, fp_weight)
        loss = (criterion(outputs, y_tensor)*weight).mean()
        loss.backward()
        optimizer.step()
    return model, scaler

# ================== 严格回测 ==================
def simulate_backtest(df_out):
    results = []
    n = len(df_out)
    for t_idx in range(n):
        if t_idx + COOLDOWN_DAYS >= n: continue
        row = df_out.iloc[t_idx]
        results.append({
            "t": t_idx,
            "price": row.get("price"),
            "decision": row.get("decision", "观望"),
            "expected_profit": row.get("expected_profit", 0)
        })
    if results:
        df_res = pd.DataFrame(results)
        cum_profit = df_res["expected_profit"].sum()
        win_rate = (df_res["expected_profit"]>0).mean()
    else:
        cum_profit = 0
        win_rate = 0
    summary = {"cum_profit": cum_profit, "win_rate": win_rate, "score": cum_profit*win_rate}
    return pd.DataFrame(results), summary

# ================== 阈值优化 ==================
def optimize_threshold(results_list, candidate_thresholds, train_ratio=0.7):
    best_thr, best_summary = None, None
    for df_out in results_list:
        n = len(df_out)
        split_idx = int(n*train_ratio)
        test_data = df_out.iloc[split_idx:]
        for thr in candidate_thresholds:
            sim_input = test_data.copy()
            if sim_input.empty: continue
            _, summary = simulate_backtest(sim_input)
            if best_summary is None or summary.get("score",0) > best_summary.get("score",0):
                best_thr, best_summary = thr, summary
    return best_thr, best_summary

# ================== 决策函数 ==================
def evaluate_goods_classification(goods_data_dict, goods_config):
    results_dict = {}
    for good_name, df in goods_data_dict.items():
        if df.empty: continue
        features = build_features(df)
        y_label = (df["price"].pct_change().shift(-1).fillna(0)>0).astype(int)

        # ===== 动态 FN/FP 权重 =====
        fn_weight = COST_MATRIX["FN"] * (1 + (df["price"].pct_change().abs().mean()))
        fp_weight = COST_MATRIX["FP"] * (1 + (df["price"].pct_change().abs().mean()))
        model, scaler = train_classifier(features, y_label, fn_weight=fn_weight, fp_weight=fp_weight)

        X_tensor = torch.tensor(scaler.transform(features), dtype=torch.float32)
        with torch.no_grad():
            prob_buy = model(X_tensor).numpy().flatten()

        decisions = []
        expected_profits = []

        for i, row in df.iterrows():
            price = row["price"]
            slope = features.loc[i, "trend_slope"]
            vol = features.loc[i, "volatility_7"]
            start_idx = max(0,i-LOOKBACK_DAYS)
            hist_min = df["price"].iloc[start_idx:i+1].min()
            hist_max = df["price"].iloc[start_idx:i+1].max()
            min_price = hist_min*MIN_PARA
            max_price = hist_max*MAX_PARA

            expected_profit = prob_buy[i]*(hist_max-price)
            expected_profits.append(expected_profit)
            exp_risk_ratio = expected_profit/(vol+1e-6)

            # ===== 动态阈值 =====
            dynamic_threshold = BUY_PROB_THRESHOLD * (1 - 0.5*vol/(vol+1e-6))
            decision = "观望"
            if price <= min_price:
                decision = "买入"
            elif price >= max_price:
                decision = "卖出"
            elif slope>0 and exp_risk_ratio>0.05 and prob_buy[i]>dynamic_threshold:
                decision = "买入"

            decisions.append(decision)

        result_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "price": df["price"],
            "trend_slope": features["trend_slope"],
            "prob_buy": prob_buy,
            "expected_profit": expected_profits,
            "decision": decisions
        })
        results_dict[good_name] = result_df

    candidate_thresholds = np.linspace(0.45,0.75,31)
    best_thr, best_summary = optimize_threshold(list(results_dict.values()), candidate_thresholds)
    logging.info(f"[evaluate_goods_classification] 最优阈值: {best_thr}, summary: {best_summary}")
    return results_dict, best_summary

# ================== 主程序 ==================
def main():
    excel_files = ["gun.xlsx"]
    goods_data_dict, goods_config = read_excel_to_goods_dict(excel_files)
    results_dict, summary = evaluate_goods_classification(goods_data_dict, goods_config)
    output_file = "purchase_decisions.xlsx"
    if results_dict:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for good_name, df_out in results_dict.items():
                sheet_name = good_name[:31] if good_name else f"Sheet_{np.random.randint(1e5)}"
                df_out.to_excel(writer, sheet_name=sheet_name, index=False)
        logging.info(f"评估结果已保存至 {output_file}")
    else:
        logging.warning("没有生成任何结果")

if __name__ == "__main__":
    main()
