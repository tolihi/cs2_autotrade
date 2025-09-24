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
BUY_PROB_THRESHOLD = 0.7
FOCUS_CATEGORY = "枪皮"
MIN_PARA = 1.1
MAX_PARA = 0.9
EXPECT_YIELD_PARA = 0.08
#小件物品的风险普遍偏高，甚至高过100%，大件普遍偏低
EXPECT_RISK_PARA_HIGH = 0.3
EXPECT_RISK_PARA_LOW = 0.6

COST_MATRIX = {
    "FN": 1.0,  # 错过买入的成本
    "FP": 5.0   # 错误买入的成本
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

    logging.info(f"已读取 {len(goods_data_dict)} 个饰品的数据")
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

    slopes = []
    for i in range(len(df)):
        start = max(0, i - SLOPE_WINDOW)
        x = np.arange(start, i + 1)
        y = df["price"].iloc[start:i + 1].values
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
        slopes.append(slope)
    features["trend_slope"] = slopes

    features["ma_7"] = df["price"].rolling(7, min_periods=1).mean()
    features["volatility_7"] = df["price"].rolling(7, min_periods=1).std().fillna(0)
    features["random_noise"] = np.random.normal(0, 0.01, len(df))
    return features.ffill().bfill()

# ================== 二元分类模型 ==================
class BuySellClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.sigmoid(self.out(x))

def train_classifier(X: pd.DataFrame, y: pd.Series, epochs=200, lr=0.0005):  #epochs——训练次数
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
        weight = torch.where(y_tensor==1, COST_MATRIX["FN"], COST_MATRIX["FP"])
        loss = (criterion(outputs, y_tensor) * weight).mean()
        loss.backward()
        optimizer.step()

    return model, scaler

# ================== 严格回测 ==================
def simulate_backtest(df_out):
    results = []
    n = len(df_out)
    logging.info(f"[simulate_backtest] 输入数据长度: {n}")

    for t_idx in range(n):
        if t_idx >= n:
            logging.warning(f"[simulate_backtest] t_idx={t_idx} 超过 df_out 长度 {n}，跳过")
            break

        try:
            row = df_out.iloc[t_idx]
        except IndexError:
            logging.error(f"[simulate_backtest] iloc[{t_idx}] 越界，df_out 长度={n}")
            break

        # 冷却期检查
        if t_idx + COOLDOWN_DAYS >= n:
            logging.debug(f"[simulate_backtest] t_idx={t_idx} 后面不足 {COOLDOWN_DAYS} 天，跳过未来计算")
            continue

        results.append({
            "t": t_idx,
            "price": row.get("price", None),
            "decision": row.get("decision", "观望"),
            "expected_profit": row.get("expected_profit", 0)
        })

    return pd.DataFrame(results), {"summary": "ok"}

def optimize_threshold(results_list, candidate_thresholds, train_ratio=0.7):
    best_thr, best_summary = None, None
    logging.info(f"[optimize_threshold] 处理 {len(results_list)} 个商品")

    for df_out in results_list:
        n = len(df_out)
        split_idx = int(n * train_ratio)
        train_data = df_out.iloc[:split_idx]
        test_data = df_out.iloc[split_idx:]

        for thr in candidate_thresholds:
            sim_input = test_data.copy()
            if sim_input.empty:
                continue
            _, summary = simulate_backtest(sim_input)

            if best_summary is None or summary.get("score", 0) > best_summary.get("score", 0):
                best_thr, best_summary = thr, summary

    return best_thr, best_summary


# ================== 决策函数 ==================
def evaluate_goods_classification(goods_data_dict, goods_config):
    """
    对每个商品进行特征构建、模型训练、决策生成，并进行阈值优化
    返回：
        results_dict: {商品名: DataFrame} 包含 timestamp, price, trend_slope, prob_buy, expected_profit, decision
        best_summary: 优化阈值的回测摘要信息
    """
    results_dict = {}

    for good_name, df in goods_data_dict.items():
        if df.empty:
            logging.warning(f"[evaluate_goods_classification] 商品 {good_name} 数据为空，跳过")
            continue

        # ================== 特征构建 ==================
        features = build_features(df)
        y_label = (df["price"].pct_change().shift(-1).fillna(0) > 0).astype(int)

        # ================== 模型训练 ==================
        model, scaler = train_classifier(features, y_label)

        # ================== 模型预测 ==================
        X_tensor = torch.tensor(scaler.transform(features), dtype=torch.float32)
        with torch.no_grad():
            prob_buy = model(X_tensor).numpy().flatten()

        decisions = []
        expected_profits = []
        expected_losses = []

        # ================== 决策逻辑 ==================
        for i, row in df.iterrows():
            price = row["price"]
            slope = features.loc[i, "trend_slope"]  #这段slope的定义有点模糊

            # 动态计算过去 LOOKBACK_DAYS 的最高/最低价
            start_idx = max(0, i - LOOKBACK_DAYS)
            hist_min = df["price"].iloc[start_idx:i+1].min()
            hist_max = df["price"].iloc[start_idx:i+1].max()
            #定义在实际判断过程中的最大最小值
            min_price = hist_min * MIN_PARA
            max_price = hist_max * MAX_PARA

            #预期收益
            expected_profit = prob_buy[i] * (max_price - price)
            expected_yield = expected_profit / price
            expected_profits.append(expected_profit)

            #预期风险
            expected_loss = prob_buy[i] * (price - min_price)
            expected_risk = expected_loss / price
            expected_losses.append(expected_loss)

            #此处购买逻辑判断错误有点大，需要修改
            #slope判断条件模糊，需更新一版新的
            decision = "观望"
            # 强制买入/卖出逻辑
            if price <= min_price:
                decision = "买入"
            elif price >= max_price:
                decision = "卖出"
            # 模型 & 趋势判断逻辑
            #上升阶段买进
            elif slope > 0 and expected_yield > EXPECT_YIELD_PARA and expected_risk < EXPECT_RISK_PARA and prob_buy[i] >= BUY_PROB_THRESHOLD:
                decision = "买入"
            #下降阶段买进
            elif slope < 0 and expected_yield > EXPECT_YIELD_PARA and expected_risk < EXPECT_RISK_PARA and prob_buy[i] >= BUY_PROB_THRESHOLD:
                decision = "买入"
            #上升阶段卖出
            elif slope > 0 and expected_yield > EXPECT_YIELD_PARA and expected_risk > EXPECT_RISK_PARA and prob_buy[i] > BUY_PROB_THRESHOLD:
                decision = "卖出"
            #下降阶段卖出
            elif slope > 0 and expected_yield > EXPECT_YIELD_PARA and expected_risk < EXPECT_RISK_PARA and prob_buy[i] > BUY_PROB_THRESHOLD:
                decision = "卖出"

            #振荡市的买入卖出判断
            #大件物品和小件物品分类判断

            else:
                decision = "观望"

            decisions.append(decision)

        # ================== 构建结果 DataFrame ==================
        result_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "price": df["price"],
            "trend_slope": features["trend_slope"],
            "prob_buy": prob_buy,
            "expected_profit": expected_profits,
            "expected_loss": expected_losses,
            "decision": decisions
        })

        results_dict[good_name] = result_df

    # ================== 阈值优化 ==================
    candidate_thresholds = np.linspace(0.45, 0.75, 31)  # 可自定义——————————阈值优化数值
    best_thr, best_summary = optimize_threshold(
        list(results_dict.values()),
        candidate_thresholds=candidate_thresholds,
        train_ratio=0.7
    )
    logging.info(f"[evaluate_goods_classification] 最优阈值: {best_thr}, summary: {best_summary}")

    return results_dict, best_summary


# ================== 主程序 ==================
def main():
    excel_files = ["gun.xlsx"]
    goods_data_dict, goods_config = read_excel_to_goods_dict(excel_files)
    results_dict, summary = evaluate_goods_classification(goods_data_dict, goods_config)

    output_file = "purchase_decisions.xlsx"
    if not results_dict:
        logging.warning("没有生成任何结果")
    else:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for good_name, df_out in results_dict.items():
                sheet_name = good_name[:31] if good_name else f"Sheet_{np.random.randint(1e5)}"
                df_out.to_excel(writer, sheet_name=sheet_name, index=False)
        logging.info(f"评估结果已保存至 {output_file}")

if __name__ == "__main__":
    main()
