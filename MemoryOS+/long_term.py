import json
import logging
import numpy as np
import faiss
from collections import deque
from typing import List, Dict, Any, Optional

try:
    from .utils import get_timestamp, get_embedding, normalize_vector, ensure_directory_exists
except ImportError:
    from utils import get_timestamp, get_embedding, normalize_vector, ensure_directory_exists

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LongTermMemory:
    """
    长期记忆类，用于存储用户画像和知识
    使用向量检索来优化知识搜索
    """
    def __init__(self, file_path: str, knowledge_capacity: int = 100,
                 embedding_model_name: str = "all-MiniLM-L6-v2", embedding_model_kwargs: dict = None, 
                 use_embedding_api: bool = False):
        """
        初始化长期记忆
        
        Args:
            file_path: 存储文件路径
            knowledge_capacity: 知识容量
            embedding_model_name: 嵌入模型名称
            embedding_model_kwargs: 嵌入模型参数
            use_embedding_api: 是否使用 API 获取嵌入
        """
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        self.knowledge_capacity = knowledge_capacity
        self.user_profiles = {}  # {user_id: {data: "profile_string", "last_updated": "timestamp"}}

        # 知识库双端队列
        self.knowledge_base = deque(maxlen=self.knowledge_capacity)  # 存储用户私人知识
        self.assistant_knowledge = deque(maxlen=self.knowledge_capacity)  # 存储助手特定知识

        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs if embedding_model_kwargs is not None else {}
        self.use_embedding_api = use_embedding_api  # 是否使用API调用embedding
        self.load()

    def update_user_profile(self, user_id: str, new_data: str, merge: bool = True):
        """
        更新用户画像
        
        Args:
            user_id: 用户 ID
            new_data: 新的画像数据
            merge: 是否合并到现有画像
        """
        if merge and user_id in self.user_profiles and self.user_profiles[user_id].get("data"):
            cur_data = self.user_profiles[user_id]["data"]
            if isinstance(cur_data, str) and isinstance(new_data, str):  # 追加
                updated_data = f"{cur_data}\n\n--- Updated on {get_timestamp()} ---\n{new_data}"
            else:  # 覆盖
                updated_data = new_data
        else:
            updated_data = new_data

        self.user_profiles[user_id] = {
            "data": updated_data,
            "last_updated": get_timestamp()
        }
        logger.info(f"LTM: Updated user profile for {user_id} (merge={merge}).")
        self.save()

    def get_raw_user_profile(self, user_id: str) -> str:
        """
        获取原始用户画像
        
        Args:
            user_id: 用户 ID
        
        Returns:
            用户画像数据
        """
        return self.user_profiles.get(user_id, {}).get("data", "None")

    def get_user_profile_data(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户画像数据
        
        Args:
            user_id: 用户 ID
        
        Returns:
            用户画像字典
        """
        return self.user_profiles.get(user_id, {})

    def add_knowledge_entry(self, knowledge_text: str, knowledge_deque: deque, type_name: str = "knowledge"):
        """
        添加知识条目
        
        Args:
            knowledge_text: 知识文本
            knowledge_deque: 知识双端队列
            type_name: 知识类型名称
        """
        if not knowledge_text or knowledge_text.strip().lower() in ["", "none", "- none", "- none."]:
            logger.info(f"LTM: Empty {type_name} received, not saving.")
            return

        # 生成知识嵌入
        vec = get_embedding(
            knowledge_text,
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        vec = normalize_vector(vec).tolist()  # 转换为列表以便JSON序列化
        
        entry = {
            "knowledge": knowledge_text,
            "timestamp": get_timestamp(),
            "knowledge_embedding": vec
        }
        
        knowledge_deque.append(entry)
        logger.info(f"LTM: Added new {type_name} entry. Total entries now: {len(knowledge_deque)}.")
        self.save()

    def add_user_knowledge(self, knowledge_text: str):
        """
        添加用户知识
        
        Args:
            knowledge_text: 知识文本
        """
        self.add_knowledge_entry(knowledge_text, self.knowledge_base, type_name="user knowledge")

    def add_assistant_knowledge(self, knowledge_text: str):
        """
        添加助手知识
        
        Args:
            knowledge_text: 知识文本
        """
        self.add_knowledge_entry(knowledge_text, self.assistant_knowledge, type_name="assistant knowledge")

    def get_user_knowledge(self) -> List[Dict[str, Any]]:
        """
        获取用户知识
        
        Returns:
            用户知识列表
        """
        return list(self.knowledge_base)

    def get_assistant_knowledge(self) -> List[Dict[str, Any]]:
        """
        获取助手知识
        
        Returns:
            助手知识列表
        """
        return list(self.assistant_knowledge)

    def _search_knowledge_deque(self, query: str, knowledge_deque: deque, 
                               threshold: float = 0.1, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索知识双端队列
        
        Args:
            query: 查询文本
            knowledge_deque: 知识双端队列
            threshold: 相似度阈值
            top_k: 返回的结果数量
        
        Returns:
            搜索结果列表
        """
        if not knowledge_deque:
            return []

        # 生成查询嵌入
        query_vec = get_embedding(
            query,
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec)
        query_arr = np.array([query_vec], dtype=np.float32)

        # 收集有效嵌入
        embeddings = []
        valid_entries = []
        for entry in knowledge_deque:
            emb = entry.get("knowledge_embedding")
            if emb is not None:
                embeddings.append(emb)
                valid_entries.append(entry)
            else:
                logger.warning(f"LTM: Entry missing 'knowledge_embedding', skipping.")

        if not embeddings:
            logger.info(f"LTM: No valid embeddings found for search.")
            return []

        # 转换为numpy数组
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # 确保嵌入维度正确
        embeddings_np = np.atleast_2d(embeddings_np)
        if embeddings_np.shape[1] != query_arr.shape[1]:
            logger.error(f"LTM: Embedding dimension mismatch. Expected {query_arr.shape[1]}, got {embeddings_np.shape[1]}.")
            return []

        # 使用FAISS进行相似度搜索
        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_np)

        distances, indices = index.search(query_arr, min(top_k, len(valid_entries)))

        # 过滤结果
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS返回-1表示无匹配
                continue
            if score >= threshold:
                results.append(valid_entries[idx])
        
        return results

    def search_user_knowledge(self, query: str, threshold: float = 0.1, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索用户知识
        
        Args:
            query: 查询文本
            threshold: 相似度阈值
            top_k: 返回的结果数量
        
        Returns:
            搜索结果列表
        """
        results = self._search_knowledge_deque(query, self.knowledge_base, threshold, top_k)
        logger.info(f"LTM: Found {len(results)} relevant user knowledge entries for query: '{query}' with threshold {threshold}.")
        return results

    def search_assistant_knowledge(self, query: str, threshold: float = 0.1, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索助手知识
        
        Args:
            query: 查询文本
            threshold: 相似度阈值
            top_k: 返回的结果数量
        
        Returns:
            搜索结果列表
        """
        results = self._search_knowledge_deque(query, self.assistant_knowledge, threshold, top_k)
        logger.info(f"LTM: Found {len(results)} relevant assistant knowledge entries for query: '{query}' with threshold {threshold}.")
        return results

    def save(self):
        """
        保存长期记忆到文件
        """
        data = {
            "user_profiles": self.user_profiles,
            "knowledge_base": list(self.knowledge_base),
            "assistant_knowledge": list(self.assistant_knowledge)
        }
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"LTM: Memory saved to {self.file_path}.")
        except IOError as e:
            logger.error(f"LTM: Error saving memory to {self.file_path}: {e}")

    def load(self):
        """
        从文件加载长期记忆
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.user_profiles = data.get("user_profiles", {})
                self.knowledge_base = deque(data.get("knowledge_base", []), maxlen=self.knowledge_capacity)
                self.assistant_knowledge = deque(data.get("assistant_knowledge", []), maxlen=self.knowledge_capacity)
            logger.info(f"LTM: Memory loaded from {self.file_path}.")
        except FileNotFoundError:
            logger.info(f"LTM: No existing memory file found at {self.file_path}. Starting fresh.")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"LTM: No existing memory found at {self.file_path} or error loading: {e}. Starting fresh.")
        except Exception as e:
            logger.error(f"LTM: Unexpected error loading memory from {self.file_path}: {e}. Starting fresh.")