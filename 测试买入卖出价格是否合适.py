import argparse  # 解析命令行参数
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import json  # 读取配置文件
from sklearn.ensemble import RandomForestClassifier  # 用随机森林作为样例模型
from sklearn.model_selection import train_test_split  # 数据集拆分
from sklearn.metrics import accuracy_score  # 模型评估

# ================= 配置 =================
DEFAULT_FILE = "all_goods_wide.xlsx"  # 默认输入文件（双击运行时会用这个）
DEFAULT_OUT = "decisions_output.csv"  # 默认输出文件
DEFAULT_CONFIG = {
    "min_expected_price": 50,   # 最低期望价（低于这个直接买入）
    "max_expected_price": 500,  # 最高期望价（高于这个直接卖出）
    "lookback_days": 90,        # 回溯天数
    "cooldown_days": 7,         # 买入冷却期（天）
    "focus_category": "枪皮",   # 只关注的分类
    "price_multiplier": 1.2     # 相对最低价倍数（用于溢价判断）
}


# ================= 加载配置 =================
def load_config(config_path=None):
    if config_path and os.path.exists(config_path):  # 如果有自定义配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG  # 否则用默认配置


# ================= 样例模型训练 =================
def train_sample_model(data):
    """ 训练一个简单模型，演示用 """
    X = data[["price"]]  # 用价格作为特征
    y = (data["price"].pct_change().shift(-1) > 0).astype(int)  # 下个时刻涨=1，否则=0
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"样例模型训练完成，测试集准确率: {acc:.2f}")
    return clf


# ================= 批量决策 =================
def evaluate_goods(data, clf, config):
    decisions = []
    goods = data["good_name"].unique()  # 获取所有饰品
    for g in goods:
        df = data[data["good_name"] == g].copy()
        df = df.sort_values("timestamp")
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
            decision = "观望"  # 溢价太高
        else:
            # 模型辅助判断
            prob = clf.predict_proba([[current_price]])[0, 1]
            decision = "买入" if prob > 0.6 else "卖出"

        decisions.append({
            "good_name": g,
            "current_price": current_price,
            "decision": decision,
            "price_ratio": price_ratio
        })
    return pd.DataFrame(decisions)


# ================= 回测模板 =================
def backtest(data, clf, config):
    """ 简单回测逻辑 """
    initial_money = 10000
    money = initial_money
    inventory = {}

    for _, row in data.iterrows():
        price = row["price"]
        name = row["good_name"]

        # 简单规则：低于 min_expected_price 就买，高于 max_expected_price 就卖
        if price < config["min_expected_price"]:
            money -= price
            inventory[name] = inventory.get(name, 0) + 1
        elif price > config["max_expected_price"] and inventory.get(name, 0) > 0:
            money += price
            inventory[name] -= 1

    # 清仓估值
    final_value = money + sum(inventory[g] * data[data["good_name"] == g]["price"].iloc[-1] for g in inventory)
    print(f"回测结果：初始资金 {initial_money}，最终资金 {final_value:.2f}，收益率 {final_value/initial_money - 1:.2%}")


# ================= 主程序 =================
def main():
    import os
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
    print(f"评估结果已保存至 {args.out}")

    # 回测
    backtest(data, clf, config)


if __name__ == "__main__":
    main()
