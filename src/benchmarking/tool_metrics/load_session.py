import json

from pathlib import Path
from typing import Any

from src.summarize_algorithms.core.models import BaseBlock, Session, ToolCallBlock


class SessionLoader:
    @staticmethod
    def load_session(path: Path | str) -> Session:
        result: list[BaseBlock] = []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

            i = 0
            while i < len(data):
                dict_block = data[i]
                block_type = dict_block["type"]

                if block_type in ("user", "system"):
                    block = BaseBlock(
                        role=block_type.upper(),
                        content=dict_block["content"]
                    )
                    result.append(block)
                    i += 1

                elif block_type == "assistant":
                    block = BaseBlock(
                        role=block_type.upper(),
                        content=dict_block["content"]
                    )
                    result.append(block)

                    tool_calls = dict_block.get("toolCalls", [])
                    if tool_calls:
                        i += 1
                        dict_block = data[i]
                        block_type = dict_block["type"]
                        tool_responses: list[dict[str, Any]] = []
                        while block_type == "tool_response":
                            tool_responses.append(dict_block)
                            i += 1
                            dict_block = data[i]
                            block_type = dict_block["type"]

                        blocks = SessionLoader._process_tool_calls(tool_calls, tool_responses)
                        result.extend(blocks)

                elif block_type == "tool_response":
                    i += 1

                else:
                    block = BaseBlock(
                        role=block_type.upper(),
                        content=str(dict_block.get("content", ""))
                    )
                    result.append(block)
                    i += 1

                return Session(result)

    @staticmethod
    def _process_tool_calls(
            tool_calls: list[dict[str, Any]],
            tool_responses: list[dict[str, Any]]
    ) -> list[ToolCallBlock]:
        blocks: list[ToolCallBlock] = []
        for call in tool_calls:
            for tool_response in tool_responses:
                if call["id"] == tool_response["id"]:
                    block = ToolCallBlock(
                        role="ASSISTANT",
                        content=tool_response["response"]["content"],
                        id=call["id"],
                        name=call["name"],
                        arguments=call["arguments"],
                        response=tool_response["result"]
                    )
                    blocks.append(block)
        return blocks
