import requests
import json
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
import os
import time

API_TOKEN = "BEDDU1T7F5R7D9U9Z0V8I7F5"
url = "https://api.csqaq.com/api/v1/info/chart"

goods = [
    {"good_id": "20653", "name": "橄榄迷彩"},
    {"good_id": "21519", "name": "灰变迷彩"},
    {"good_id": "10365", "name": "迷踪秘境"},
    {"good_id": "16498", "name": "钢铁三角洲"},
    {"good_id": "62", "name": "精英之作"},
]

OUTPUT_FILE = "all_goods_wide.csv"
FIXED_TIMES = [dt_time(10,0), dt_time(22,0)]  # 每日抓取时间

# ================== 数据抓取 ==================
def fetch_price_history(good_id):
    payload = json.dumps({
        "good_id": good_id,
        "key": "sell_price",
        "platform": 1,
        "period": "90",   # 最近90天
        "style": "all_style"
    })
    headers = {
        'ApiToken': API_TOKEN,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            res_json = response.json()
            if res_json.get('code') == 200:
                data = res_json.get('data', {})
                timestamps = data.get('timestamp', [])
                prices = data.get('main_data', [])
                volumes = data.get('num_data', [])
                if timestamps and prices:
                    df = pd.DataFrame({
                        "timestamp": [datetime.fromtimestamp(ts/1000) for ts in timestamps],
                        "price": prices,
                        "volume": volumes if volumes else [0]*len(prices)
                    })
                    df['daily_change'] = df['price'].pct_change() * 100
                    return df
        print(f"请求饰品 {good_id} 返回数据异常或为空")
        return pd.DataFrame()
    except Exception as e:
        print(f"获取饰品 {good_id} 数据失败: {e}")
        return pd.DataFrame()

# ================== CSV 更新 ==================
def update_wide_csv(goods_data_dict):
    # 读取已有 CSV
    if os.path.exists(OUTPUT_FILE):
        wide_df = pd.read_csv(OUTPUT_FILE, parse_dates=['timestamp'])
    else:
        wide_df = pd.DataFrame({"timestamp": []})

    all_rows = []
    for name, df in goods_data_dict.items():
        if df.empty:
            continue
        df['date'] = df['timestamp'].dt.date
        grouped = df.groupby('date')
        rows = []
        for day, group in grouped:
            group = group.sort_values('timestamp').reset_index(drop=True)  # ✅ 重置索引
            for ft in FIXED_TIMES:
                target_dt = datetime.combine(day, ft)
                now = datetime.now()
                if target_dt > now:
                    continue  # 还没到该固定时间，不记录
                closest_row = group.iloc[(group['timestamp'] - target_dt).abs().argsort()[0]]
                row = {
                    "timestamp": target_dt,
                    f"{name}_price": closest_row['price'],
                    f"{name}_volume": closest_row['volume'],
                    f"{name}_change": closest_row['daily_change']
                }
                rows.append(row)
        all_rows.append(pd.DataFrame(rows))

    if not all_rows:
        print("没有可更新的数据")
        return

    # 合并所有饰品数据按 timestamp 对齐
    combined_df = pd.DataFrame({"timestamp": sorted({row['timestamp'] for df in all_rows for _, row in df.iterrows()})})
    for i, name in enumerate(goods_data_dict.keys()):
        df = all_rows[i]
        combined_df = pd.merge(combined_df, df, on='timestamp', how='left')

        # 插入空列隔开
        combined_df[f"{name}_sep"] = ""

    # 合并已有 CSV，去重
    if not wide_df.empty:
        combined = pd.concat([wide_df, combined_df], ignore_index=True)
        combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    else:
        combined = combined_df

    combined.sort_values('timestamp', inplace=True)
    combined.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"{OUTPUT_FILE} 已更新，总行数: {len(combined)}")

# ================== 主循环 ==================
while True:
    print(f"--- 数据刷新 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    goods_data = {}
    for good in goods:
        df = fetch_price_history(good["good_id"])
        goods_data[good["name"]] = df
        if not df.empty:
            print(f"{good['name']} 最新价格: {df['price'].iloc[-1]:.2f} RMB")

    update_wide_csv(goods_data)

    # 计算到下一个固定时间的秒数
    now = datetime.now()
    next_times = [datetime.combine(now.date(), t) for t in FIXED_TIMES if datetime.combine(now.date(), t) > now]
    if not next_times:
        next_times = [datetime.combine(now.date() + timedelta(days=1), FIXED_TIMES[0])]
    wait_seconds = (min(next_times) - now).total_seconds()
    print(f"等待 {int(wait_seconds)} 秒后抓取下一条数据...\n")
    time.sleep(wait_seconds)
