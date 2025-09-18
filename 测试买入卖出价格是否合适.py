import argparse  # 解析命令行参数
import os
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import json  # 读取配置文件
import logging  # 更好的日志管理
from sklearn.ensemble import RandomForestClassifier  # 用随机森林作为样例模型
from sklearn.model_selection import train_test_split  # 数据集拆分
from sklearn.metrics import accuracy_score  # 模型评估

# ================= 配置 =================
DEFAULT_FILE = "枪皮.xlsx"  # 默认输入文件
DEFAULT_OUT = "decisions_output.csv"  # 默认输出文件

DEFAULT_CONFIG = {
    "min_expected_price": 50,   # 最低期望价
    "max_expected_price": 500,  # 最高期望价
    "lookback_days": 90,        # 回溯天数
    "cooldown_days": 7,         # 买入冷却期
    "focus_category": "枪皮",   # 只关注的分类
    "price_multiplier": 1.2     # 溢价判断倍数
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
"""
指定日志的最低等级。
Python 的日志等级从低到高是：
DEBUG < INFO < WARNING < ERROR < CRITICAL
设成 INFO，就表示：
INFO, WARNING, ERROR, CRITICAL 会被输出；
DEBUG 不会显示。
这样能避免太多调试信息刷屏
"""
)




# ================= 加载配置文件 =================
def load_config(config_path: str | None = None) -> dict:
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG


# ================= 样例模型训练 =================
def train_sample_model(data: pd.DataFrame) -> RandomForestClassifier:
    """ 训练一个简单模型，演示用 """
    # 基础特征
    X = pd.DataFrame({
        "price": data["price"],
        "rolling_mean_5": data["price"].rolling(5).mean(),
        "rolling_std_5": data["price"].rolling(5).std()
    }).fillna(method="bfill")

    # 标签：未来是否上涨
    y = (data["price"].pct_change().shift(-1) > 0).astype(int).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"样例模型训练完成，测试集准确率: {acc:.2f}")
    return clf


# ================= 批量决策 =================
def evaluate_goods(data: pd.DataFrame, clf: RandomForestClassifier, config: dict) -> pd.DataFrame:
    results = []
    grouped = data.sort_values("timestamp").groupby("good_name")

    for g, df in grouped:
        if df.empty:
            continue

        current_price = df["price"].iloc[-1]
        min_price = df["price"].tail(config["lookback_days"]).min()
        price_ratio = current_price / min_price if min_price > 0 else 1

        # 规则判断
        if current_price < config["min_expected_price"]:
            decision = "买入"
        elif current_price > config["max_expected_price"]:
            decision = "卖出"
        elif price_ratio > config["price_multiplier"]:
            decision = "观望"
        else:
            # 模型辅助判断
            prob = clf.predict_proba(np.array([[current_price,
                                                df["price"].rolling(5).mean().iloc[-1],
                                                df["price"].rolling(5).std().iloc[-1]]]))[0, 1]
            decision = "买入" if prob > 0.6 else "卖出"

        results.append({
            "good_name": g,
            "current_price": current_price,
            "decision": decision,
            "price_ratio": price_ratio
        })

    return pd.DataFrame(results)


# ================= 回测模板 =================
def backtest(data: pd.DataFrame, clf: RandomForestClassifier, config: dict) -> None:
    """ 简单回测逻辑 """
    initial_money = 10000
    money = initial_money
    inventory = {}

    for _, row in data.iterrows():
        price = row["price"]
        name = row["good_name"]

        if price < config["min_expected_price"]:
            money -= price
            inventory[name] = inventory.get(name, 0) + 1
        elif price > config["max_expected_price"] and inventory.get(name, 0) > 0:
            money += price
            inventory[name] = max(0, inventory[name] - 1)

    # 清仓估值
    final_value = money + sum(
        inventory[g] * data[data["good_name"] == g]["price"].iloc[-1]
        for g in inventory
    )
    logging.info(
        f"回测结果：初始资金 {initial_money}，最终资金 {final_value:.2f}，收益率 {final_value/initial_money - 1:.2%}"
    )


# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser(description="测试买入卖出价格是否合适")
    parser.add_argument("--file", type=str, help="输入文件路径（Excel/CSV）", default=DEFAULT_FILE)
    parser.add_argument("--config", type=str, help="配置文件路径", default=None)
    parser.add_argument("--out", type=str, help="输出文件路径", default=DEFAULT_OUT)
    args = parser.parse_args()

    # 读取数据
    if args.file.endswith(".xlsx"):
        data = pd.read_excel(args.file)
    else:
        data = pd.read_csv(args.file)

    # 加载配置
    config = load_config(args.config)

    # 训练模型
    clf = train_sample_model(data)

    # 批量评估
    results = evaluate_goods(data, clf, config)
    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    logging.info(f"评估结果已保存至 {args.out}")

    # 回测
    backtest(data, clf, config)


if __name__ == "__main__":
    main()
