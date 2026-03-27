"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import logging
import re
import time
from typing import Optional, Dict, Any, List

import httpx
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError

from ..config import Config

logger = logging.getLogger(__name__)

# 网络瞬断重试配置
_RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError, httpx.ConnectError, httpx.RemoteProtocolError)
_MAX_RETRIES = 2
_RETRY_DELAY = 2  # 首次重试等待秒数，后续翻倍


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=Config.LLM_TIMEOUT,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数（默认使用配置中的 LLM_MAX_TOKENS）
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        # 如果未指定 max_tokens，使用配置中的默认值
        effective_max_tokens = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
            except _RETRYABLE_EXCEPTIONS as e:
                print(f"LLM 连接异常: {e}")
                last_error = e
                wait = _RETRY_DELAY * (2 ** attempt)
                logger.warning(f"LLM 连接失败（第 {attempt + 1}/{_MAX_RETRIES} 次），{wait}s 后重试: {e}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
                continue
            except Exception as e:
                # 如果是因为 response_format 导致的错误，尝试不使用 response_format 重试
                error_str = str(e).lower()
                if response_format and ("response_format" in error_str or 
                                        "json_object" in error_str or
                                        "unsupported" in error_str or
                                        "400" in error_str or
                                        "500" in error_str):
                    # 移除 response_format 后重试
                    kwargs.pop("response_format", None)
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            content = response.choices[0].message.content
            # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
            content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
            return content
        raise last_error
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数（默认使用配置中的 LLM_MAX_TOKENS）

        Returns:
            解析后的JSON对象
        """
        try:
            # 首先尝试使用 response_format 参数（OpenAI 原生支持）
            response = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            # 如果失败，尝试不使用 response_format 重试
            error_str = str(e).lower()
            if ("response_format" in error_str or 
                "json_object" in error_str or
                "unsupported" in error_str or
                "400" in error_str or
                "500" in error_str):
                # 不使用 response_format，依赖系统提示词中的 JSON 格式要求
                response = self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                raise
        
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
