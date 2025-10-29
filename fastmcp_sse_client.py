import asyncio
import json
import logging
import os

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, model_name: str, url: str, api_key: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=url)

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """
        发送消息给LLM并获取响应
        :param messages: 发送的消息列表
        :return: LLM的响应
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content


class ChatSession:
    def __init__(self, llm_client: LLMClient, mcp_session: Client) -> None:
        self.llm_client: LLMClient = llm_client
        self.mcp_session: Client = mcp_session

    async def process_llm_response(self, llm_response: str) -> str:
        """
        处理LLM响应，解析工具调用并执行
        """
        try:
            # 移除可能得markdown格式
            if llm_response.startswith("```json"):
                llm_response = llm_response.strip("```json").strip("```").strip()
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                # 检查工具列表是否包含该工具
                tools = await self.mcp_session.list_tools()
                if any(tool.name == tool_call["tool"] for tool in tools):
                    try:
                        result = await self.mcp_session.call_tool(tool_call["tool"], tool_call["arguments"])
                        return f"调用了{tool_call['tool']}工具，结果为{result.content[0].text}"
                    except Exception as e:
                        error_msg = f"调用工具时出错：{str(e)}"
                        logging.error(error_msg)
                        return error_msg
                return f"没有该工具：{tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            # 非JSON格式直接返回LLM响应
            return llm_response

    async def start(self, system_message) -> None:
        """
        聊天会话主循环
        """
        messages = [{"role": "system", "content": system_message}]
        while True:
            try:
                # 获取用户输入
                user_input = input("用户：").strip().lower()
                if user_input in ["quit", "exit", "退出"]:
                    print("用户已退出")
                    break
                messages.append({"role": "user", "content": user_input})
                # LLM初始响应
                llm_response = self.llm_client.get_response(messages)
                print("助手：", llm_response)
                # 处理可能的工具调用
                result = await self.process_llm_response(llm_response)
                # 循环处理LLM响应中的工具调用
                while result != llm_response:
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "system", "content": result})
                    # 将工具调用结果添加到消息列表中，获取新响应
                    llm_response = self.llm_client.get_response(messages)
                    print("助手：", result)
                    result = await self.process_llm_response(llm_response)

                messages.append({"role": "assistant", "content": llm_response})
            except KeyboardInterrupt:
                print("用户已退出")
                break


async def main():
    async with Client("http://127.0.0.1:3001/sse") as session:
        # 初始化LLM客户端
        llm_client = LLMClient(model_name=os.getenv("MODEL"), api_key=os.getenv("OPENAI_API_KEY"),
                               url=os.getenv("BASE_URL"))
        # 获取可用工具列表并格式化为提示词一部分
        tools = await session.list_tools()
        dict_list = [tool.__dict__ for tool in tools]
        tools_description = json.dumps(dict_list, ensure_ascii=False)
        # 系统提示，指导LLM如何使用工具
        system_message = f'''
        你是一个智能助手，严格遵循以下协议返回响应。

        可用工具：
        {tools_description}

        相应规则：
        1、当需要计算时，返回严格符合以下格式的JSON：
        {{
            "tool": "tool_name",
            "arguments": {{
                "argument_name": "argument_value"                
            }}
        }}

        返回的JSON禁止包含以下内容：
        - Markdown标记
        - 自然语言解释（如“结果为：”）
        - 格式化后的数据（必须保持原始数据精度）
        - 单位符号

        校验流程：
        参数数量与工具参数一致
        数值类型的参数必须为数字
        JSON格式必须正确

        正确示例：
        用户：一份订单88.5元，卖235份多少钱？
        响应：
        {{
            "tool": "multiply",
            "arguments": {{
                "a": 88.5,
                "b": 235
            }}
        }}

        2、收到工具的响应后：
        - 将原始数据转化为自然、对话式的回应
        - 保持回复简洁
        - 聚焦于最相关的信息
        - 可以使用用户问题中的合适上下文
        - 避免简单重复原始数据
        '''
        chat_session = ChatSession(llm_client, session)
        await chat_session.start(system_message)


if __name__ == '__main__':
    asyncio.run(main())