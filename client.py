import asyncio
import os
import json
from typing import Optional, List
from contextlib import AsyncExitStack
from datetime import datetime
import re
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("未找到OPENAI_API_KEY环境变量，请在.env文件中设置。")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None

    async def connect_to_server(self, server_script_path: str):
        is_python = os.path.splitext(server_script_path)[-1] == ".py"
        is_node = os.path.splitext(server_script_path)[-1] == ".js"
        if not is_python and not is_node:
            raise ValueError("服务器脚本仅支持python或typescript文件")
        # 根据文件后缀选择启动命令
        command = "python" if is_python else "node"
        # MCP服务器参数，包括启动命令、脚本路径参数、环境变量（None为使用默认）
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        # 建立MCP stdio通信
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        # 创建MCP客户端会话对象
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        # 初始化会话
        await self.session.initialize()
        # 获取工具列表
        response = await self.session.list_tools()
        tools = response.tools
        print("MCP服务器支持以下工具:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        # 获取工具列表
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
            } for tool in response.tools
        ]
        # 提取用户查询关键词，生成文件名
        keyword_match = re.search(r'(关于|分析|查询|搜索|查看)([^的\s，。？、\n]+)', query)
        keyword = keyword_match.group(2) if keyword_match else "分析对象"
        safe_keyword = re.sub(r'[\\/:*?"<>|]', "", keyword)[:20]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f"sentiment_{safe_keyword}_{timestamp}.txt"
        txt_path = os.path.join("sentiment_report", txt_filename)
        # 更新查询，将文件名添加到原始查询中，使大模型调用工具时可以考虑到文件名
        new_query = query.strip() + f" [txt_filename:{txt_filename}] [txt_path:{txt_path}]"
        messages = [
            {
                "role": "user",
                "content": new_query
            }
        ]
        tool_plan = await self.plan_tool_use(new_query, available_tools)
        tool_outputs = {}
        # 依次执行工具
        for step in tool_plan:
            tool_name = step["name"]
            tool_args = step["arguments"]
            # 参数动态绑定，使得后续工具可以把前面工具的返回结果当参数
            for key, val in tool_args.items():
                if isinstance(val, str) and val.startswith("{{") and val.endswith("}}"):
                    ref_key = val.strip("{} ")
                    resolved_val = tool_outputs.get(ref_key, None)
                    tool_args[key] = resolved_val
            # 注入统一的文件名和路径
            if tool_name == "analyze_sentiment" and "filename" not in tool_args:
                tool_args["filename"] = txt_filename
            if tool_name == "send_email_with_attachment" and "attachment_path" not in tool_args:
                tool_args["attachment_path"] = txt_path

            result = await self.session.call_tool(tool_name, tool_args)
            tool_outputs[tool_name] = result.content[0].text  # 记录工具返回结果
            messages.append({"role": "tool", "tool_call_id": tool_name, "content": result.content[0].text})
        # 使用大模型生成汇总结果
        final_result = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        final_output = final_result.choices[0].message.content

        # 获取文件名辅助函数
        def clean_filename(text: str) -> str:
            text = text.strip()
            text = re.sub(r'[\\/:*?"<>|]', '', text)
            return text[:50]

        # 存储查询和大模型回复
        safe_filename = clean_filename(query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_filename}_{timestamp}.txt"
        output_dir = "./llm_output"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"用户输入：{query}\n")
            f.write(f"模型回复：{final_output}\n")
        print(f"对话记录已保存到文件：{file_path}")
        return final_output

    async def chat_loop(self):
        # 初始化提示信息
        print("MCP客户端已启动，输入 'quit' 退出")
        while True:
            try:
                query = input("请输入内容：").strip()
                if query == "quit":
                    break
                response = await self.process_query(query)
                print(response)
            except Exception as e:
                print(f"发生错误：{str(e)}")

    async def plan_tool_use(self, query:str, tools:List[dict]) -> List[dict]:
        print("\n提交给大模型的工具有：")
        print(json.dumps(tools, esure_ascii=False, indent=2))
        tool_list_text = "\n".join([f"--{tool['function']['name']}: {tool['function']['description']}" for tool in tools])
        # 构造全局提示
        system_prompt = {
            "role": "system",
            "content": (
                "你是一个智能任务规划助手，用户会给出自然语言请求。\n"
                "你只能从以下工具中选择（严格使用工具名称）：\n"
                f"{tool_list_text}\n"
                "如果需多个工具串联使用，后续步骤中可以使用 {{上一步工具名}} 占位。\n"
                "返回格式：JSON数组，每个元素是一个对象，包含 name 和 arguments 两个字段，name 是工具名，arguments 是工具参数。\n"""
                "不要返回自然语言，不要使用未列出的工具名。\n"
            )
        }

        plan_messages = [
            system_prompt,
            {"role": "user", "content": query}
        ]
        plan_response = self.client.chat.completions.create(
            model=self.model,
            messages=plan_messages,
            tools=tools,
            tool_choice="none",
        )
        # 提取模型响应中的JSON内容
        content = plan_response.choices[0].message.content.strip()
        match = re.search(r"(?:json)?\s*([\s\S]+?)\s*", content)
        if match:
            json_text = match.group(1)
        else:
            json_text = content
        # 解析json内容并返回
        try:
            plan = json.loads(json_text)
            return plan if isinstance(plan, list) else []
        except Exception as e:
            print(f"工具调用链规划失败，原始返回为：{content}")
            return []

    async def clean_up(self):
        await self.exit_stack.aclose()


async def main():
    server_script_path = "server.py"
    client = MCPClient()
    try:
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.clean_up()

if __name__ == '__main__':
    asyncio.run(main())