# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import requests
import pandas as pd
import time

if __name__ == "__main__":
    parse_bilibili_json("raw_data.json")

def fetch_bilibili_comments(oid, pages=3):
    """
    抓取B站指定视频的评论
    :param oid: 视频的av号（或者是B站API需要的oid）
    :param pages: 抓取的页数
    """
    url = "https://api.bilibili.com/x/v2/reply/wbi/main?oid=116278186478338&type=1&mode=3&pagination_str=%7B%22offset%22:%22%22%7D&plat=1&seek_rpid=&web_location=1315875&w_rid=4f0fb123247e421a1443667fcb07f221&wts=1774667302"
    my_cookie = "buvid3=B4BE05F3-B4E8-B227-6049-2E1C1A978A3D06248infoc; b_nut=1774237206; bsource=search_google; _uuid=45675E74-D210B-88D6-A421-A2106D10EED591007209infoc; buvid_fp=df992c6359f068f11f881401a48d1e6b; home_feed_column=4; buvid4=24BF6D40-F2ED-7641-C263-0760DB1A374007378-026032311-lcxStzaU9aFNIf1sfXBj+A%3D%3D; browser_resolution=736-750; CURRENT_FNVAL=4048; CURRENT_QUALITY=0; rpdid=|(J~k)m||k0J'u~~RYYY)Y); bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzQ5MjY0NjQsImlhdCI6MTc3NDY2NzIwNCwicGx0IjotMX0.OXGxTYFSlrxqzsFWruXHLZWEWAg0gNbrRmlnmUgdkRc; bili_ticket_expires=1774926404; SESSDATA=8fd0bb0b%2C1790219278%2Cd0d40%2A32CjD8gX7gFfI4ki1JWgChdvfdanso5o0_EmtSBCb7mpIGN7rA7xmcCzAK2GOir5y7XC0SVjRCQWkzY24xMjdFUHo5UUQ1d09GY2oyd2o2NFdRcGFrNkN6RmlfYUpDTHdDQlpwZ2pSUDBtRVJfT2pNdFhYaUVVck9IN2JpTWRLLXJLVVVUZGZudE9RIIEC; bili_jct=931ab3b04438745b82c66f345d83fe1f; DedeUserID=188736391; DedeUserID__ckMd5=8a21a55e7bc77e5f; sid=5xglifci; theme-tip-show=SHOWED; b_lsid=CB58B200_19D327385E4"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": f"https://www.bilibili.com/video/av{oid}/",  # 必须带上最后的斜杠
        "Cookie": my_cookie,
        "Origin": "https://www.bilibili.com"
    }
    # 模拟浏览器请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/"
    }

    all_comments = []

    for page in range(1, pages + 1):
        # 这里的参数根据B站API要求调整
        params = {
            "oid": oid,
            "type": 1,
            "mode": 3,
            "pagination_str": '{"offset":""}' if page == 1 else f'{{"offset":"{page * 20}"}}',
            "plat": 1
        }

        print(f"正在抓取第 {page} 页...")

        try:
            response = requests.get(url, params=params, headers=headers)
            data = response.json()

            # 【调试】如果抓不到，看看 B 站到底返回了什么代码
            if data['code'] != 0:
                print(f"B站服务器返回错误！错误码: {data['code']}，信息: {data['message']}")
                print("提示：如果错误码是 -101，说明你的 Cookie 没填对或失效了。")
                break

            if data['data'] and data['data']['replies']:
                replies = data['data']['replies']
                for r in replies:
                    all_comments.append({
                        "user": r['member']['uname'],
                        "comment": r['content']['message'],
                        "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['ctime']))
                    })
                print(f"成功抓取第 {page} 页，当前累计 {len(all_comments)} 条。")
            else:
                print("本页没有更多评论了。")
                break

        except Exception as e:
            print(f"程序运行出错: {e}")

        time.sleep(2)

        return all_comments

        # 爬虫要有礼貌，休息一下，防止被封IP
        time.sleep(2)

    return all_comments


# 示例：抓取某个视频的评论（oid需要根据实际视频查找）
# 比如某个星铁视频的oid是 1152643597
comments_list = fetch_bilibili_comments(oid="116278186478338", pages=5)

# 保存为CSV文件，方便后续交给AI分析
if comments_list:
    df = pd.DataFrame(comments_list)
    df.to_csv("game_comments.csv", index=False, encoding="utf-8-sig")
    print(f"\n全部抓取完成！共保存 {len(df)} 条评论到 game_comments.csv")
else:
    print("\n最终抓取结果为空，请检查上述错误信息。")