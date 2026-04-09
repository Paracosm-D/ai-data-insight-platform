import json
import pandas as pd
import time


def parse_bilibili_json(file_path):
    # 1. 读取本地 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 定位到评论列表 (B站接口数据通常在 data -> replies 下)
    # 注意：有的接口层级是 data['replies']，有的是 data['data']['replies']
    # 我们加一个简单的判断来增强代码鲁棒性
    replies = []
    if 'data' in data and 'replies' in data['data']:
        replies = data['data']['replies']
    elif 'replies' in data:
        replies = data['replies']

    if not replies:
        print("未在 JSON 中找到评论内容，请检查文件内容是否完整。")
        return

    # 3. 提取关键字段
    extracted_data = []
    for r in replies:
        item = {
            "user": r['member']['uname'],  # 用户名
            "comment": r['content']['message'],  # 评论正文
            "like_count": r['like'],  # 点赞数（运营很看重这个指标）
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['ctime']))  # 时间转换
        }
        extracted_data.append(item)

    # 4. 存入 CSV
    df = pd.DataFrame(extracted_data)
    output_file = "game_comments.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"解析成功！已从 JSON 中提取 {len(df)} 条评论并保存至 {output_file}")
    print(df.head())  # 预览前几行


if __name__ == "__main__":
    parse_bilibili_json("raw_data.json")