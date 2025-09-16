import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import time

matplotlib.rc("font", family='SimHei')   # 使用黑体
matplotlib.rc("axes", unicode_minus=False)

API_TOKEN = "BEDDU1T7F5R7D9U9Z0V8I7F5"
url = "https://api.csqaq.com/api/v1/info/chart"

payload = json.dumps({
    "good_id": "7246",
    "key": "sell_price",
    "platform": 1,
    "period": "30",   #1095
    "style": "all_style"
})

headers = {
    'ApiToken': API_TOKEN,
    'Content-Type': 'application/json'
}

# 定时循环
interval = 10   # 单位秒，比如600秒 = 10分钟

while True:
    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('code') == 200:
                data = response_data.get('data', {})
                timestamps = data.get('timestamp', [])
                prices = data.get('main_data', [])
                volumes = data.get('num_data', [])

                dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]

                # 清空并绘制
                plt.clf()
                plt.plot(dates, prices, label="Sell Price", color="blue")

                if volumes:
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    ax2.bar(dates, volumes, alpha=0.3, color="orange", label="Volume")
                    ax2.set_ylabel("成交量 / 上架数量", color="orange")

                plt.title("饰品价格走势")
                plt.xlabel("日期")
                plt.ylabel("价格 (RMB)")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.pause(1)   # 动态刷新图像，不会卡死

            else:
                print(f"请求失败: {response_data.get('msg')}")
        else:
            print(f"HTTP错误: {response.status_code}")

    except Exception as e:
        print("发生错误:", e)

    # 等待下次循环
    time.sleep(interval)
