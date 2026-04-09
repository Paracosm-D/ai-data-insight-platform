import pandas as pd
import time
import requests
import json

API_KEY = "你的DeepSeek API Key"
BASE_URL = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "请解释multi-armed bandit"}
    ]
}

# 2. 定义系统提示词 (System Prompt) —— 这是展现你运营思维的核心！
SYSTEM_PROMPT = """
你是一个资深的游戏数据运营分析师。
你的任务是分析玩家的社区评论，并提取结构化数据，帮助业务团队快速了解舆情。

请对输入的玩家评论进行以下维度的分析，并严格以 JSON 格式输出：
1. sentiment (情感倾向): "正向", "中性", "负向"
2. category (问题分类): 从以下选项中选择最贴合的一项 -> ["角色与剧情", "福利与活动", "BUG与优化", "战斗数值与抽卡", "其他"]
3. keywords (核心槽点/亮点): 提取1-3个关键词，如 "吃大保底", "福利抠门", "立绘好看"

输出格式示例：
{
    "sentiment": "负向",
    "category": "战斗数值与抽卡",
    "keywords": ["吃大保底", "爆率低"]
}
"""


def analyze_comment_with_llm(comment_text):
    """调用 LLM 对单条评论进行分析"""
    try:
        response = client.chat.completions.create(
            model= "deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请分析这条玩家评论：\n{comment_text}"}
            ],
            response_format={"type": "json_object"},  # 强制要求输出 JSON，便于后续数据处理
            temperature=0.3  # 降低温度，保证分类的一致性
        )

        # 解析返回的 JSON 字符串
        result_json = json.loads(response.choices[0].message.content)
        return result_json

    except Exception as e:
        print(f"分析出错: {e}")
        return {"sentiment": "未知", "category": "未知", "keywords": []}


def main():
    # 3. 读取本地收集好的玩家评论数据
    # 假设你有一个 csv 文件，包含一列名为 'comment'
    print("正在加载评论数据...")
    try:
        df = pd.read_csv("game_comments.csv")
    except FileNotFoundError:
        # 如果没有本地文件，这里生成几条测试数据用于演示
        print("未找到 game_comments.csv，使用测试数据...")
        data = {
            "comment": [
                "这期活动送的星琼太少了吧，抠搜的，而且剧情做得像白开水一样。",
                "新出的角色技能机制很有趣，大招特效直接拉满，必须抽爆！",
                "服了，打个深渊经常闪退，还吃我大保底，赶紧修复BUG啊！"
            ]
        }
        df = pd.DataFrame(data)

    # 创建新的列来存储 AI 分析结果
    sentiments = []
    categories = []
    keywords_list = []

    # 4. 批量处理评论
    print(f"开始使用 AI 分析舆情，共 {len(df)} 条数据...")
    for index, row in df.iterrows():
        comment = row['comment']
        print(f"正在处理第 {index + 1} 条...")

        analysis_result = analyze_comment_with_llm(comment)

        sentiments.append(analysis_result.get('sentiment', '未知'))
        categories.append(analysis_result.get('category', '未知'))
        # 将列表形式的关键词转换为逗号分隔的字符串，方便存入 CSV
        keywords_list.append(", ".join(analysis_result.get('keywords', [])))

        # 适度休眠，防止触发 API 频率限制
        time.sleep(0.5)

    # 5. 将结果合并回 DataFrame
    df['情感倾向'] = sentiments
    df['反馈分类'] = categories
    df['核心关键词'] = keywords_list

    # 6. 导出结构化数据
    output_filename = "analyzed_game_feedback.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n分析完成！结构化数据已保存至 {output_filename}")

    # 打印前几行看看效果
    print("\n分析结果预览：")
    print(df.head())


if __name__ == "__main__":
    main()