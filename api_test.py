import requests
import json

# 配置你的API Token
API_TOKEN = "BEDDU1T7F5R7D9U9Z0V8I7F5"  # 重要：请替换成你从CSQAQ平台获取的真实API Token

# 价格走势接口URL
url = "https://api.csqaq.com/api/v1/info/chart"

# 构造请求负载 (Payload)，以爪子刀（★） | 外表生锈 (战痕累累)为例，id是7246:cite[2]
payload = json.dumps({
    "good_id": "7246",  # 饰品ID
    "key": "sell_price",  # 价格类型，例如出售价格
    "platform": 1,  # 平台标识符（例如1可能代表BUFF平台，请以文档为准）
    "period": "1095",  # 查询周期，例如1095天（约3年）:cite[2]
    "style": "all_style"  # 样式（具体含义请参考官方文档）
})

# 设置请求头
headers = {
    'ApiToken': API_TOKEN,  # 在此处进行身份验证:cite[1]:cite[2]
    'Content-Type': 'application/json'  # 指定请求体内容类型为JSON
}

try:
    # 发送POST请求
    response = requests.post(url, headers=headers, data=payload)

    # 检查HTTP响应状态码
    if response.status_code == 200:
        # 解析返回的JSON数据
        response_data = response.json()

        # 检查API业务状态码（根据文档，200通常表示成功）:cite[1]
        if response_data.get('code') == 200:
            # 请求成功，处理返回的价格走势数据 (通常在 response_data['data'] 中)
            print("请求成功！")
            price_history_data = response_data.get('data', {})
            # 接下来你可以对price_history_data进行处理和可视化
            print(json.dumps(price_history_data, indent=2, ensure_ascii=False))  # 美化打印输出
        else:
            # API返回了业务逻辑错误
            print(f"请求失败，业务状态码: {response_data.get('code')}, 错误信息: {response_data.get('msg')}")
    else:
        # HTTP请求本身出现问题（如网络错误、服务器内部错误等）
        print(f"HTTP请求失败，状态码: {response.status_code}")

except requests.exceptions.RequestException as e:
    # 处理网络请求异常（如连接超时、无法连接等）
    print(f"网络请求发生异常: {e}")
except json.JSONDecodeError as e:
    # 处理JSON解析异常
    print(f"JSON解析失败: {e}")