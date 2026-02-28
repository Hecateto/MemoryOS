import inspect
import json
import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple

import numpy as np
import openai
from sentence_transformers import SentenceTransformer

try:
    from . import prompts  # 尝试相对导入
except ImportError:
    import prompts  # 回退到绝对导入
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_reasoning_model_output(text: Optional[str]) -> Optional[str]:
    """
    清理推理模型输出中的<think>标签
    适配推理模型（如o1系列）的输出格式
    
    Args:
        text: 输入文本
    
    Returns:
        清理后的文本
    """
    if not text:
        return text

    import re
    # 移除<think>...</think>标签及其内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 清理可能产生的多余空白行
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    # 移除开头和结尾的空白
    cleaned_text = cleaned_text.strip()

    return cleaned_text


# ---- OpenAI Client ----
class OpenAIClient:
    """
    OpenAI 客户端封装，用于处理 LLM 调用
    """
    def __init__(self, api_key: str, base_url: Optional[str] = None, max_workers: int = 5):
        """
        初始化 OpenAI 客户端
        
        Args:
            api_key: OpenAI API 密钥
            base_url: API 基础 URL，默认为官方地址
            max_workers: 线程池最大工作线程数
        """
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        # 显式传递参数给客户端构造函数
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def chat_completion(self, model: str, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        调用 OpenAI 聊天完成 API
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大令牌数
        
        Returns:
            模型响应内容
        """
        logger.info(f"Calling OpenAI API. Model: {model}")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_content = response.choices[0].message.content.strip()
            # 自动清理推理模型的<think>标签
            cleaned_content = clean_reasoning_model_output(raw_content)
            return cleaned_content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # 错误处理
            return "Error: Could not get response from LLM."

    def chat_completion_async(self, model: str, messages: List[Dict[str, str]], 
                             temperature: float = 0.7, max_tokens: int = 2000):
        """
        异步版本的 chat_completion
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大令牌数
        
        Returns:
            Future 对象
        """
        return self.executor.submit(self.chat_completion, model, messages, temperature, max_tokens)

    def batch_chat_completion(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        并行处理多个 LLM 请求
        
        Args:
            requests: 请求列表，每个请求包含 model、messages、temperature、max_tokens 等键
        
        Returns:
            响应结果列表
        """
        futures = []
        for req in requests:
            future = self.chat_completion_async(
                model=req.get("model", "gpt-4o-mini"),
                messages=req["messages"],
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens", 2000)
            )
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch completion: {e}")
                results.append("Error: Could not get response from LLM.")

        return results

    def shutdown(self):
        """
        关闭线程池
        """
        self.executor.shutdown(wait=True)


# ---- Parallel Processing Utilities ----
def run_parallel_tasks(tasks: List[Callable], max_workers: int = 3) -> List[Any]:
    """
    并行执行任务列表
    
    Args:
        tasks: 可调用函数列表
        max_workers: 最大工作线程数
    
    Returns:
        任务执行结果列表
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel task: {e}")
                results.append(None)
        return results


# ---- Basic Utilities ----
def get_timestamp() -> str:
    """
    获取当前时间戳
    
    Returns:
        格式化的时间戳字符串
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def generate_id(prefix: str = "id") -> str:
    """
    生成唯一 ID
    
    Args:
        prefix: ID 前缀
    
    Returns:
        唯一 ID 字符串
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def ensure_directory_exists(path: str):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 文件路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---- Embedding Utilities ----
_model_cache = {}  # 模型缓存
_embedding_cache = {}  # embedding 缓存
_embedding_api_client = None  # API embedding 的 OpenAI 客户端


def set_embedding_api_client(client: OpenAIClient):
    """
    设置用于 API embedding 的 OpenAI 客户端
    
    Args:
        client: OpenAIClient 实例，用于调用 embedding API
    """
    global _embedding_api_client
    _embedding_api_client = client
    logger.info(f"Embedding API client set. Base URL: {client.base_url}")


def _get_embedding_via_api(text: str, model_name: str, use_cache: bool = True) -> np.ndarray:
    """
    通过 OpenAI 兼容 API 获取 embedding 向量
    
    Args:
        text: 输入文本
        model_name: API 上的模型名称，如 'Qwen/Qwen3-Embedding-8B'
        use_cache: 是否使用内存缓存
    
    Returns:
        embedding 向量 (numpy array)
    """
    global _embedding_api_client, _embedding_cache

    if _embedding_api_client is None:
        raise ValueError("API embedding client not set. Call set_embedding_api_client() first or use use_api=False.")

    cache_key = f"api::{model_name}::{hash(text)}"
    if use_cache and cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    logger.info(f"Calling Embedding API. Model: {model_name}")
    try:
        response = _embedding_api_client.client.embeddings.create(
            model=model_name,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error calling Embedding API: {e}")
        raise

    if use_cache:
        _embedding_cache[cache_key] = embedding
        # 缓存大小控制
        if len(_embedding_cache) > 10000:
            keys_to_remove = list(_embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                try:
                    del _embedding_cache[key]
                except KeyError:
                    pass
            logger.info("Cleaned embedding cache to prevent memory overflow")

    return embedding


def _get_valid_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    过滤函数签名中有效的关键字参数
    
    Args:
        func: 目标函数
        kwargs: 关键字参数字典
    
    Returns:
        过滤后的关键字参数字典
    """
    try:
        sig = inspect.signature(func)
        param_keys = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in param_keys}
    except (ValueError, TypeError):
        # 对于签名检查不明确的函数，返回原始 kwargs
        return kwargs


def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2", 
                  use_cache: bool = True, use_api: bool = False, **kwargs) -> np.ndarray:
    """
    获取文本的 embedding 向量
    支持多种主流模型，能自动适应不同库的调用方式
    - SentenceTransformer模型: e.g., 'all-MiniLM-L6-v2', 'Qwen/Qwen3-Embedding-0.6B'
    - FlagEmbedding模型: e.g., 'BAAI/bge-m3'
    - API模型: 通过 OpenAI 兼容 API 调用，如 'Qwen/Qwen3-Embedding-8B'
    
    Args:
        text: 输入文本
        model_name: 模型名称（本地或API上的模型名称）
        use_cache: 是否使用内存缓存
        use_api: 如果为 True，则通过 OpenAI 兼容 API 调用 embedding 服务
                需要先调用 set_embedding_api_client() 设置客户端
        kwargs: 传递给模型构造函数或encode方法的额外参数
                - for Qwen: `model_kwargs`, `tokenizer_kwargs`, `prompt_name="query"`
                - for BGE-M3: `use_fp16=True`, `max_length=8192`
    
    Returns:
        文本的 embedding 向量 (numpy array)
    """
    # 如果使用 API 调用
    if use_api:
        return _get_embedding_via_api(text, model_name, use_cache)

    # --- 本地模型逻辑 ---
    model_config_key = json.dumps({"model_name": model_name, **kwargs}, sort_keys=True)

    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]

    # --- Model Loading ---
    model_init_key = json.dumps(
        {"model_name": model_name, **{k: v for k, v in kwargs.items() if k not in ['batch_size', 'max_length']}},
        sort_keys=True)
    
    if model_init_key not in _model_cache:
        logger.info(f"Loading model: {model_name}...")
        if 'bge-m3' in model_name.lower():
            try:
                from FlagEmbedding import BGEM3FlagModel
                init_kwargs = _get_valid_kwargs(BGEM3FlagModel.__init__, kwargs)
                logger.debug(f"-> Using BGEM3FlagModel with init kwargs: {init_kwargs}")
                _model_cache[model_init_key] = BGEM3FlagModel(model_name, **init_kwargs)
            except ImportError:
                raise ImportError("Please install FlagEmbedding: 'pip install -U FlagEmbedding' to use bge-m3 model.")
        else:  # Default handler for SentenceTransformer-based models (like Qwen, all-MiniLM, etc.)
            try:
                from sentence_transformers import SentenceTransformer
                init_kwargs = _get_valid_kwargs(SentenceTransformer.__init__, kwargs)
                logger.debug(f"-> Using SentenceTransformer with init kwargs: {init_kwargs}")
                _model_cache[model_init_key] = SentenceTransformer(model_name, **init_kwargs)
            except ImportError:
                raise ImportError(
                    "Please install sentence-transformers: 'pip install -U sentence-transformers' to use this model.")

    model = _model_cache[model_init_key]

    # --- Encoding ---
    embedding = None
    if 'bge-m3' in model_name.lower():
        encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
        logger.debug(f"-> Encoding with BGEM3FlagModel using kwargs: {encode_kwargs}")
        result = model.encode([text], **encode_kwargs)
        embedding = result['dense_vecs'][0]
    else:  # Default to SentenceTransformer-based models
        encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
        logger.debug(f"-> Encoding with SentenceTransformer using kwargs: {encode_kwargs}")
        embedding = model.encode([text], **encode_kwargs)[0]

    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        _embedding_cache[cache_key] = embedding
        # 缓存大小控制
        if len(_embedding_cache) > 10000:
            keys_to_remove = list(_embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                try:
                    del _embedding_cache[key]
                except KeyError:
                    pass
            logger.info("Cleaned embedding cache to prevent memory overflow")

    return embedding


def clear_embedding_cache():
    """
    清空 embedding 缓存
    """
    global _embedding_cache
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    归一化向量
    
    Args:
        vec: 输入向量
    
    Returns:
        归一化后的向量
    """
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# ---- Time Decay Function ----
def compute_time_decay(event_timestamp_str: str, current_timestamp_str: str, tau_hours: float = 24) -> float:
    """
    计算时间衰减因子
    
    Args:
        event_timestamp_str: 事件时间戳字符串
        current_timestamp_str: 当前时间戳字符串
        tau_hours: 衰减参数（小时）
    
    Returns:
        时间衰减因子
    """
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t_event = datetime.strptime(event_timestamp_str, fmt)
        t_current = datetime.strptime(current_timestamp_str, fmt)
        delta_hours = (t_current - t_event).total_seconds() / 3600.0
        return float(np.exp(-delta_hours / tau_hours))
    except ValueError:  # 处理时间戳无效的情况
        return 0.1  # 默认低时效性


# ---- LLM-based Utility Functions ----

def gpt_summarize_dialogs(dialogs: List[Dict[str, Any]], client: OpenAIClient, model: str = "gpt-4o-mini") -> str:
    """
    使用 LLM 总结对话
    
    Args:
        dialogs: 对话列表
        client: OpenAIClient 实例
        model: 模型名称
    
    Returns:
        对话总结
    """
    dialog_text = "\n".join(
        [f"User: {d.get('user_input', '')} Assistant: {d.get('agent_response', '')}" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.SUMMARIZE_DIALOGS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.SUMMARIZE_DIALOGS_USER_PROMPT.format(dialog_text=dialog_text)}
    ]
    logger.info("Calling LLM to generate topic summary...")
    return client.chat_completion(model=model, messages=messages)


def gpt_generate_multi_summary(text: str, client: OpenAIClient, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    使用 LLM 生成多主题摘要
    
    Args:
        text: 输入文本
        client: OpenAIClient 实例
        model: 模型名称
    
    Returns:
        包含输入文本和摘要列表的字典
    """
    import re
    messages = [
        {"role": "system", "content": prompts.MULTI_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.MULTI_SUMMARY_USER_PROMPT.format(text=text)}
    ]
    logger.info("Calling LLM to generate multi-topic summary...")
    response_text = client.chat_completion(model=model, messages=messages)

    logger.info(f"Raw multi-summary response: {response_text}")

    try:
        json_match = re.search(r'\[.*]', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*}', response_text, re.DOTALL)

        if json_match:
            clean_json_str = json_match.group(0)
        else:
            clean_json_str = response_text

        summaries = json.loads(clean_json_str)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse multi-summary JSON: {response_text}")
        summaries = []  # 返回空列表
    logger.info(f"Parsed multi-summary: {summaries}")
    return {"input": text, "summaries": summaries}


def gpt_user_profile_analysis(dialogs: List[Dict[str, Any]], client: OpenAIClient, 
                              model: str = "gpt-4o-mini", existing_user_profile: str = "None") -> str:
    """
    分析和更新用户个性画像
    结合现有画像和新对话，直接输出更新后的完整画像
    
    Args:
        dialogs: 对话列表
        client: OpenAIClient 实例
        model: 模型名称
        existing_user_profile: 现有用户画像
    
    Returns:
        更新后的用户画像
    """
    conversation = "\n".join([
                                 f"User: {d.get('user_input', '')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response', '')} (Timestamp: {d.get('timestamp', '')})"
                                 for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.PERSONALITY_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.PERSONALITY_ANALYSIS_USER_PROMPT.format(
            conversation=conversation,
            existing_user_profile=existing_user_profile
        )}
    ]
    logger.info("Calling LLM for user profile analysis and update...")
    result_text = client.chat_completion(model=model, messages=messages)
    return result_text.strip() if result_text else "None"


def gpt_knowledge_extraction(dialogs: List[Dict[str, Any]], client: OpenAIClient, 
                             model: str = "gpt-4o-mini") -> Dict[str, str]:
    """
    从对话中提取用户私人数据和助手知识
    
    Args:
        dialogs: 对话列表
        client: OpenAIClient 实例
        model: 模型名称
    
    Returns:
        包含私人数据和助手知识的字典
    """
    conversation = "\n".join([
                                 f"User: {d.get('user_input', '')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response', '')} (Timestamp: {d.get('timestamp', '')})"
                                 for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.KNOWLEDGE_EXTRACTION_USER_PROMPT.format(
            conversation=conversation
        )}
    ]
    logger.info("Calling LLM for knowledge extraction...")
    result_text = client.chat_completion(model=model, messages=messages)

    private_data = "None"
    assistant_knowledge = "None"

    try:
        if "【User Private Data】" in result_text:
            private_data_start = result_text.find("【User Private Data】") + len("【User Private Data】")
            if "【Assistant Knowledge】" in result_text:
                private_data_end = result_text.find("【Assistant Knowledge】")
                private_data = result_text[private_data_start:private_data_end].strip()

                assistant_knowledge_start = result_text.find("【Assistant Knowledge】") + len("【Assistant Knowledge】")
                assistant_knowledge = result_text[assistant_knowledge_start:].strip()
            else:
                private_data = result_text[private_data_start:].strip()
        elif "【Assistant Knowledge】" in result_text:
            assistant_knowledge_start = result_text.find("【Assistant Knowledge】") + len("【Assistant Knowledge】")
            assistant_knowledge = result_text[assistant_knowledge_start:].strip()

    except Exception as e:
        logger.error(f"Error parsing knowledge extraction: {e}. Raw result: {result_text}")

    return {
        "private": private_data if private_data else "None",
        "assistant_knowledge": assistant_knowledge if assistant_knowledge else "None"
    }


# Keep the old function for backward compatibility, but mark as deprecated
def gpt_personality_analysis(dialogs: List[Dict[str, Any]], client: OpenAIClient, 
                             model: str = "gpt-4o-mini", known_user_traits: str = "None") -> Dict[str, str]:
    """
    DEPRECATED: Use gpt_user_profile_analysis and gpt_knowledge_extraction instead.
    This function is kept for backward compatibility only.
    
    Args:
        dialogs: 对话列表
        client: OpenAIClient 实例
        model: 模型名称
        known_user_traits: 已知用户特征
    
    Returns:
        包含用户画像、私人数据和助手知识的字典
    """
    # Call the new functions
    profile = gpt_user_profile_analysis(dialogs, client, model, known_user_traits)
    knowledge_data = gpt_knowledge_extraction(dialogs, client, model)

    return {
        "profile": profile,
        "private": knowledge_data["private"],
        "assistant_knowledge": knowledge_data["assistant_knowledge"]
    }


def gpt_update_profile(old_profile: str, new_analysis: str, client: OpenAIClient, 
                       model: str = "gpt-4o-mini") -> str:
    """
    更新用户画像
    
    Args:
        old_profile: 旧用户画像
        new_analysis: 新分析结果
        client: OpenAIClient 实例
        model: 模型名称
    
    Returns:
        更新后的用户画像
    """
    messages = [
        {"role": "system", "content": prompts.UPDATE_PROFILE_SYSTEM_PROMPT},
        {"role": "user",
         "content": prompts.UPDATE_PROFILE_USER_PROMPT.format(old_profile=old_profile, new_analysis=new_analysis)}
    ]
    logger.info("Calling LLM to update user profile...")
    return client.chat_completion(model=model, messages=messages)


def gpt_extract_theme(answer_text: str, client: OpenAIClient, model: str = "gpt-4o-mini") -> str:
    """
    提取主题
    
    Args:
        answer_text: 回答文本
        client: OpenAIClient 实例
        model: 模型名称
    
    Returns:
        提取的主题
    """
    messages = [
        {"role": "system", "content": prompts.EXTRACT_THEME_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.EXTRACT_THEME_USER_PROMPT.format(answer_text=answer_text)}
    ]
    logger.info("Calling LLM to extract theme...")
    return client.chat_completion(model=model, messages=messages)


# ---- Functions from dynamic_update.py (to be used by Updater class) ----
def check_conversation_continuity(previous_page: Optional[Dict[str, Any]], 
                                 current_page: Dict[str, Any], 
                                 client: OpenAIClient, 
                                 model: str = "gpt-4o-mini") -> bool:
    """
    检查对话连续性
    
    Args:
        previous_page: 上一页对话
        current_page: 当前页对话
        client: OpenAIClient 实例
        model: 模型名称
    
    Returns:
        是否连续
    """
    prev_user = previous_page.get("user_input", "") if previous_page else ""
    prev_agent = previous_page.get("agent_response", "") if previous_page else ""

    user_prompt = prompts.CONTINUITY_CHECK_USER_PROMPT.format(
        prev_user=prev_user,
        prev_agent=prev_agent,
        curr_user=current_page.get("user_input", ""),
        curr_agent=current_page.get("agent_response", "")
    )
    messages = [
        {"role": "system", "content": prompts.CONTINUITY_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat_completion(model=model, messages=messages, temperature=0.0, max_tokens=10)
    return response.strip().lower() == "true"


def generate_page_meta_info(last_page_meta: Optional[str], 
                           current_page: Dict[str, Any], 
                           client: OpenAIClient, 
                           model: str = "gpt-4o-mini") -> str:
    """
    生成页面元信息
    
    Args:
        last_page_meta: 上一页元信息
        current_page: 当前页对话
        client: OpenAIClient 实例
        model: 模型名称
    
    Returns:
        生成的元信息
    """
    current_conversation = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
    user_prompt = prompts.META_INFO_USER_PROMPT.format(
        last_meta=last_page_meta if last_page_meta else "None",
        new_dialogue=current_conversation
    )
    messages = [
        {"role": "system", "content": prompts.META_INFO_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    return client.chat_completion(model=model, messages=messages, temperature=0.3, max_tokens=100).strip()