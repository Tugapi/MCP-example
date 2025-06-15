import os
import json
import smtplib
from datetime import datetime
from email.message import EmailMessage

import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

mcp = FastMCP("NewsServer")


@mcp.tool()
async def search_google_news(keywords: str) -> str:
    """
    使用 Serper API 根据关键词搜素新闻。
    :param keywords: 搜索关键词
    :return: JSON字符串，包含前5条新闻标题、描述、链接
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("未找到SERPER_API_KEY环境变量，请在.env文件中设置。")
    # 设置请求参数并发送请求
    url = "https://google.serper.dev/news/"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "q": keywords
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        data = response.json()
    # 检查数据，按格式返回前5条新闻
    if "news" in data:
        return "未获取到搜索结果。"
    articles = [
        {
            "title": item.get("title"),
            "desc": item.get("snippet"),
            "url": item.get("link")
        } for item in data["news"][:5]
    ]
    # 将搜索结果以JSON格式保存到本地
    output_dir = "./google_news"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"google_news_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    return (
        f"已获取与[{keywords}]相关的前5条Google新闻：\n"
        f"{json.dumps(articles, ensure_ascii=False, indent=2)}\n"
        f"结果已保存到文件：{file_path}"
    )


@mcp.tool()
async def analyze_sentiment(text: str, file_name: str) -> str:
    """
    对传入文本进行情感分析，把结果保存到指定名称的md文件。
    :param text: 文本内容
    :param file_name: 给定文件名
    :return: 完整文件路径（用于邮件发送）
    """
    # 情感分析功能用LLM实现，需调用OpenAI API
    openai_api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL")
    client = OpenAI(api_key=openai_api_key, base_url=base_url)
    # 情感分析提示词
    prompt = f"请根据以下新闻内容进行情绪倾向分析，并说明原因：\n\n{text}"
    # 调用LLM API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    result = response.choices[0].message.content
    # 保存结果
    markdown = f"""# 新闻情绪倾向分析报告
    **分析时间：** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ---
    ## 新闻内容：
    {text}
    ---
    ## 分析结果：
    {result}
    """
    output_dir = "./sentiment_report"
    os.makedirs(output_dir, exist_ok=True)
    if not file_name:
        file_name = f"sentiment_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return file_path


@mcp.tool()
async def send_email_with_attachment(to: str, subject: str, body: str, file_name: str) -> str:
    """
    发送带有附件的邮件。
    :param to: 收件人邮箱地址
    :param subject: 邮件标题
    :param body: 邮件正文
    :param file_name: 作为附件的文件名
    :return: 邮件发送状态说明
    """
    # 获取并配置SMTP相关信息
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_pass = os.getenv("SENDER_PASS")
    file_path = os.path.abspath(os.path.join("./sentiment_report", file_name))
    if not os.path.exists(file_path):
        return "附件文件不存在。"
    # 创建邮件对象
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to
    msg.set_content(body)
    # 添加附件并发送邮件
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)
    except Exception as e:
        return f"添加附件失败：{str(e)}"
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_pass)
            server.send_message(msg)
        return f"邮件已成功发送给{to}，附件为{file_path}"
    except Exception as e:
        return f"邮件发送失败：{str(e)}"

if __name__ == '__main__':
    mcp.run(transport="stdio")