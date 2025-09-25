import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 参数设置
LOOKBACK_DAYS = 90
SLOPE_WINDOW = 30
COOLDOWN_DAYS = 7
BUY_PROB_THRESHOLD = 0.5
FOCUS_CATEGORY = "枪皮"
MIN_PARA = 1.05
MAX_PARA = 0.95
EXPECT_YIELD_PARA = 0.05
EXPECT_RISK_PARA_HIGH = 0.4
EXPECT_RISK_PARA_LOW = 0.7
MINUTE_PRICE = 500

# 惩罚系数
PENALTY_SLOPE_DOWN = 1.02
PENALTY_SLOPE_UP = 0.98
PENALTY_SLOPE_BUMPY = 1

# 交易成本参数
TRANSACTION_FEE = 0.001
STOP_LOSS_RATIO = 0.05
TAKE_PROFIT_RATIO = 0.15

COST_MATRIX = {
    "FN": 1.5,
    "FP": 3.0
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
            df = remove_outliers(df, 'price')

            goods_data_dict[sheet_name] = df
            min_price = df["price"].rolling(LOOKBACK_DAYS, min_periods=1).min().iloc[-1] * MIN_PARA
            max_price = df["price"].rolling(LOOKBACK_DAYS, min_periods=1).max().iloc[-1] * MAX_PARA
            goods_config[sheet_name] = {"min_expected_price": min_price, "max_expected_price": max_price}

    logging.info(f"已读取 {len(goods_data_dict)} 个饰品的数据")
    return goods_data_dict, goods_config


def remove_outliers(df, column, threshold=3):
    if len(df) == 0:
        return df
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] >= mean - threshold * std) & (df[column] <= mean + threshold * std)]


# ================== 特征构建 ==================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    features = pd.DataFrame(index=df.index)
    features["price"] = df["price"]

    # 滞后特征
    features["price_lag1"] = df["price"].shift(1).bfill()
    features["price_lag3"] = df["price"].shift(3).bfill()
    features["price_lag7"] = df["price"].shift(7).bfill()

    # 技术指标
    features["price_ratio"] = df["price"] / df["price"].rolling(LOOKBACK_DAYS, min_periods=1).min()
    features["price_change_1d"] = df["price"].pct_change().fillna(0)
    features["price_change_7d"] = df["price"].pct_change(7).fillna(0)

    # 移动平均线
    features["ma_7"] = df["price"].rolling(7, min_periods=1).mean()
    features["ma_30"] = df["price"].rolling(30, min_periods=1).mean()
    features["ma_ratio"] = features["ma_7"] / features["ma_30"].replace(0, 1)

    # 波动率
    features["volatility_7"] = df["price"].rolling(7, min_periods=1).std().fillna(0)
    features["volatility_30"] = df["price"].rolling(30, min_periods=1).std().fillna(0)

    # 相对强弱指数(RSI)
    features["rsi"] = calculate_rsi(df["price"], 14)

    # 趋势斜率
    slopes = []
    for i in range(len(df)):
        start = max(0, i - SLOPE_WINDOW)
        x = np.arange(start, i + 1)
        y = df["price"].iloc[start:i + 1].values
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
        slopes.append(slope)
    features["trend_slope"] = slopes

    # 划痕与新品价格比率
    if "price_scratched" in df.columns and "price_new" in df.columns:
        features["scratched_new_ratio"] = df["price_scratched"] / df["price_new"].replace(0, 1)
    else:
        features["scratched_new_ratio"] = 1.0

    return features.ffill().bfill()


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.1)
    return 100 - (100 / (1 + rs)).fillna(50)


# ================== 二元分类模型 ==================
class BuySellClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        return self.sigmoid(self.out(x))


def train_classifier(X: pd.DataFrame, y: pd.Series, epochs=150, lr=0.001):
    class_counts = y.value_counts()
    logging.info(f"类别分布: {class_counts.to_dict()}")

    if len(class_counts) == 2:
        class_weight = torch.tensor([class_counts[1] / class_counts[0]], dtype=torch.float32)
    else:
        class_weight = torch.tensor([1.0], dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    model = BuySellClassifier(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float('inf')
    patience = 20
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)

        weight = torch.where(y_train_tensor == 1, COST_MATRIX["FN"], COST_MATRIX["FP"])
        loss = (criterion(outputs, y_train_tensor) * weight).mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_preds = (val_outputs > 0.5).float()
            val_acc = accuracy_score(y_val_tensor.numpy(), val_preds.numpy())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"早停在第 {epoch + 1} 轮，最佳验证损失: {best_val_loss:.6f}")
                break

    model.load_state_dict(best_model)

    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        val_outputs = model(X_val_tensor)

        train_preds = (train_outputs > 0.5).float()
        val_preds = (val_outputs > 0.5).float()

        train_acc = accuracy_score(y_train_tensor.numpy(), train_preds.numpy())
        train_precision = precision_score(y_train_tensor.numpy(), train_preds.numpy(), zero_division=0)
        train_recall = recall_score(y_train_tensor.numpy(), train_preds.numpy(), zero_division=0)

        val_acc = accuracy_score(y_val_tensor.numpy(), val_preds.numpy())
        val_precision = precision_score(y_val_tensor.numpy(), val_preds.numpy(), zero_division=0)
        val_recall = recall_score(y_val_tensor.numpy(), val_preds.numpy(), zero_division=0)

    logging.info(f"模型训练完成 - 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
    logging.info(f"训练精确率: {train_precision:.4f}, 训练召回率: {train_recall:.4f}")
    logging.info(f"验证精确率: {val_precision:.4f}, 验证召回率: {val_recall:.4f}")

    return model, scaler


# ================== 回测系统 ==================
def simulate_backtest(df_out):
    results = []
    n = len(df_out)
    logging.info(f"[simulate_backtest] 输入数据长度: {n}")

    in_position = False
    entry_price = 0.0
    entry_time = None
    total_return = 0.0
    trade_count = 0
    winning_trades = 0

    for t_idx in range(n):
        try:
            row = df_out.iloc[t_idx]
        except IndexError:
            logging.error(f"[simulate_backtest] iloc[{t_idx}] 越界，df_out 长度={n}")
            break

        current_price = row["price"]
        decision = row["decision"]
        timestamp = row["timestamp"]

        result = {
            "t": t_idx,
            "timestamp": timestamp,
            "price": current_price,
            "decision": decision,
            "expected_profit": row.get("expected_profit", 0),
            "expected_loss": row.get("expected_loss", 0),
            "in_position": in_position,
            "return": 0.0,
            "cumulative_return": total_return
        }

        if in_position:
            current_return = (current_price - entry_price) / entry_price - TRANSACTION_FEE

            if current_return >= TAKE_PROFIT_RATIO:
                decision = "止盈卖出"
                result["decision"] = decision
                in_position = False
                total_return += current_return
                trade_count += 1
                winning_trades += 1
                result["return"] = current_return
            elif current_return <= -STOP_LOSS_RATIO:
                decision = "止损卖出"
                result["decision"] = decision
                in_position = False
                total_return += current_return
                trade_count += 1
                result["return"] = current_return

        if decision == "买入" and not in_position:
            if entry_time is None or (timestamp - entry_time) >= timedelta(days=COOLDOWN_DAYS):
                in_position = True
                entry_price = current_price
                entry_time = timestamp
                result["entry_price"] = entry_price

        elif decision == "卖出" and in_position:
            trade_return = (current_price - entry_price) / entry_price - TRANSACTION_FEE
            in_position = False
            total_return += trade_return
            trade_count += 1
            if trade_return > 0:
                winning_trades += 1
            result["return"] = trade_return
            result["exit_price"] = current_price

        result["cumulative_return"] = total_return
        results.append(result)

    win_rate = winning_trades / trade_count if trade_count > 0 else 0
    summary = {
        "total_return": total_return,
        "trade_count": trade_count,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "avg_return_per_trade": total_return / trade_count if trade_count > 0 else 0
    }

    return pd.DataFrame(results), summary


def optimize_threshold(results_list, candidate_thresholds, train_ratio=0.7):
    best_thr, best_summary = None, None
    best_score = -float('inf')
    logging.info(f"[optimize_threshold] 处理 {len(results_list)} 个商品")

    for df_out in results_list:
        n = len(df_out)
        if n == 0:
            continue

        split_idx = int(n * train_ratio)
        train_data = df_out.iloc[:split_idx]
        test_data = df_out.iloc[split_idx:]

        for thr in candidate_thresholds:
            sim_input = test_data.copy()
            sim_input["decision"] = sim_input.apply(
                lambda row: "买入" if row["prob_buy"] >= thr else row["decision"], axis=1
            )

            if sim_input.empty:
                continue

            _, summary = simulate_backtest(sim_input)

            score = summary["total_return"] * 0.7 + summary["win_rate"] * 0.3

            if score > best_score:
                best_score = score
                best_thr, best_summary = thr, summary

    return best_thr, best_summary


# ================== 决策函数 ==================
def evaluate_goods_classification(goods_data_dict, goods_config):
    results_dict = {}

    for good_name, df in goods_data_dict.items():
        if df.empty:
            logging.warning(f"[evaluate_goods_classification] 商品 {good_name} 数据为空，跳过")
            continue

        # 特征构建
        features = build_features(df)

        # 确保基础数据长度一致
        base_length = len(df)
        if len(features) != base_length:
            logging.warning(
                f"[evaluate_goods_classification] 特征长度({len(features)})与数据长度({base_length})不一致，统一调整为最短长度")
            base_length = min(len(features), len(df))
            df = df.iloc[:base_length].reset_index(drop=True)
            features = features.iloc[:base_length].reset_index(drop=True)

        # 生成标签
        future_return = df["price"].pct_change().shift(-1).fillna(0)
        y_label = (future_return > 0).astype(int)

        # 模型训练
        model, scaler = train_classifier(features, y_label)

        # 模型预测 - 确保预测结果长度正确
        X_scaled = scaler.transform(features)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            prob_buy = model(X_tensor).numpy().flatten()

        # 确保prob_buy长度正确
        if len(prob_buy) != base_length:
            logging.warning(f"prob_buy长度({len(prob_buy)})与基础长度({base_length})不一致，进行调整")
            if len(prob_buy) > base_length:
                prob_buy = prob_buy[:base_length]
            else:
                prob_buy = np.pad(prob_buy, (0, base_length - len(prob_buy)), mode='edge')

        # 初始化列表，确保长度一致
        decisions = [None] * base_length
        expected_profits = [0.0] * base_length
        expected_losses = [0.0] * base_length

        # 决策逻辑
        for i in range(base_length):
            try:
                row = df.iloc[i]
                price = row["price"]
                slope = features.iloc[i]["trend_slope"]
                rsi = features.iloc[i]["rsi"]
            except Exception as e:
                logging.error(f"处理索引 {i} 时出错: {str(e)}")
                decisions[i] = "观望"
                continue

            # 计算历史高低价
            start_idx = max(0, i - LOOKBACK_DAYS)
            hist_min = df["price"].iloc[start_idx:i + 1].min()
            hist_max = df["price"].iloc[start_idx:i + 1].max()
            min_price = hist_min * MIN_PARA
            max_price = hist_max * MAX_PARA

            # 计算预期收益和风险
            potential_gain = max(0, max_price - price)
            potential_loss = max(0, price - min_price)

            expected_profit = prob_buy[i] * potential_gain
            expected_yield = expected_profit / price if price > 0 else 0

            expected_loss = prob_buy[i] * potential_loss
            expected_risk = expected_loss / price if price > 0 else 0

            expected_profits[i] = expected_profit
            expected_losses[i] = expected_loss

            # 基础决策
            decision = "观望"

            # 强制买入/卖出逻辑
            if price <= min_price:
                decision = "买入"
            elif price >= max_price:
                decision = "卖出"
            else:
                # 风险参数
                risk_param = EXPECT_RISK_PARA_HIGH if price >= MINUTE_PRICE else EXPECT_RISK_PARA_LOW

                # 惩罚系数
                if slope > 0:
                    penalty = PENALTY_SLOPE_UP
                elif slope < 0:
                    penalty = PENALTY_SLOPE_DOWN
                else:
                    penalty = PENALTY_SLOPE_BUMPY

                # 买入条件
                buy_conditions = [
                    expected_yield > EXPECT_YIELD_PARA * penalty * 0.8,
                    expected_risk < risk_param * penalty * 1.2,
                    prob_buy[i] >= BUY_PROB_THRESHOLD * penalty * 0.8,
                    expected_risk < expected_yield * 1.2
                ]

                # 卖出条件
                sell_conditions = [
                    expected_yield > EXPECT_YIELD_PARA * penalty,
                    expected_risk > risk_param * penalty,
                    expected_risk > expected_yield
                ]

                # RSI条件
                if rsi > 70:
                    sell_conditions.append(True)
                elif rsi < 30:
                    buy_conditions.append(True)

                if all(buy_conditions):
                    decision = "买入"
                elif all(sell_conditions):
                    decision = "卖出"

            decisions[i] = decision

        # 检查所有数组长度
        logging.info(f"构建结果DataFrame前的长度检查:")
        logging.info(f"timestamp: {len(df['timestamp'])}, price: {len(df['price'])}")
        logging.info(f"trend_slope: {len(features['trend_slope'])}, rsi: {len(features['rsi'])}")
        logging.info(f"prob_buy: {len(prob_buy)}, expected_profit: {len(expected_profits)}")
        logging.info(f"expected_loss: {len(expected_losses)}, decision: {len(decisions)}")

        # 强制所有数组长度一致
        try:
            # 提取特征列并确保长度
            trend_slope = features["trend_slope"].values[:base_length]
            rsi = features["rsi"].values[:base_length]

            # 确保所有数组长度一致
            assert len(df["timestamp"]) == base_length, f"timestamp长度不匹配"
            assert len(df["price"]) == base_length, f"price长度不匹配"
            assert len(trend_slope) == base_length, f"trend_slope长度不匹配"
            assert len(rsi) == base_length, f"rsi长度不匹配"
            assert len(prob_buy) == base_length, f"prob_buy长度不匹配"
            assert len(expected_profits) == base_length, f"expected_profits长度不匹配"
            assert len(expected_losses) == base_length, f"expected_losses长度不匹配"
            assert len(decisions) == base_length, f"decisions长度不匹配"
        except AssertionError as e:
            logging.error(f"数据长度不一致: {str(e)}，进行强制修正")
            # 对不匹配的数组进行截断或填充
            trend_slope = features["trend_slope"].values[:base_length] if len(
                features["trend_slope"]) >= base_length else np.pad(features["trend_slope"].values,
                                                                    (0, base_length - len(features["trend_slope"])),
                                                                    mode='edge')
            rsi = features["rsi"].values[:base_length] if len(features["rsi"]) >= base_length else np.pad(
                features["rsi"].values, (0, base_length - len(features["rsi"])), mode='edge')
            prob_buy = prob_buy[:base_length] if len(prob_buy) >= base_length else np.pad(prob_buy, (0,
                                                                                                     base_length - len(
                                                                                                         prob_buy)),
                                                                                          mode='edge')
            expected_profits = expected_profits[:base_length] + [0.0] * (base_length - len(expected_profits)) if len(
                expected_profits) < base_length else expected_profits[:base_length]
            expected_losses = expected_losses[:base_length] + [0.0] * (base_length - len(expected_losses)) if len(
                expected_losses) < base_length else expected_losses[:base_length]
            decisions = decisions[:base_length] + ["观望"] * (base_length - len(decisions)) if len(
                decisions) < base_length else decisions[:base_length]

        # 构建结果DataFrame
        result_df = pd.DataFrame({
            "timestamp": df["timestamp"].values[:base_length],
            "price": df["price"].values[:base_length],
            "trend_slope": trend_slope,
            "rsi": rsi,
            "prob_buy": prob_buy,
            "expected_profit": expected_profits,
            "expected_loss": expected_losses,
            "decision": decisions
        })

        results_dict[good_name] = result_df

    # 阈值优化
    candidate_thresholds = np.linspace(0.45, 0.75, 31)
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
                backtest_results, backtest_summary = simulate_backtest(df_out)

                sheet_name = good_name[:31] if good_name else f"Sheet_{np.random.randint(1e5)}"
                backtest_results.to_excel(writer, sheet_name=sheet_name, index=False)

                summary_sheet = f"{sheet_name}_summary"
                summary_df = pd.DataFrame([backtest_summary])
                summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)

                logging.info(f"{good_name} 回测结果: {backtest_summary}")

        logging.info(f"评估结果已保存至 {output_file}")


if __name__ == "__main__":
    main()
