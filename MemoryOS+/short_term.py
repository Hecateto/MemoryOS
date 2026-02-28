import json
import logging
from collections import deque
from typing import List, Dict, Any, Optional

try:
    from .utils import get_timestamp, ensure_directory_exists
except ImportError:
    from utils import get_timestamp, ensure_directory_exists

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
短期记忆模块，使用 FIFO 队列实现，并支持持久化存储
"""
class ShortTermMemory:
    """
    短期记忆类，用于存储最近的对话历史
    使用 FIFO 队列实现，当达到最大容量时会自动淘汰最旧的条目
    """
    def __init__(self, file_path: str, max_capacity: int = 10):
        """
        初始化短期记忆
        
        Args:
            file_path: 存储文件路径
            max_capacity: 最大容量
        """
        self.file_path = file_path
        self.max_capacity = max_capacity
        ensure_directory_exists(self.file_path)
        self.memory = deque(maxlen=max_capacity)
        self.load()

    def add_qa_pair(self, qa_pair: Dict[str, Any]):
        """
        添加问答对到短期记忆
        
        Args:
            qa_pair: 问答对字典，包含 user_input、agent_response 等字段
        """
        if 'timestamp' not in qa_pair or not qa_pair['timestamp']:
            qa_pair["timestamp"] = get_timestamp()

        self.memory.append(qa_pair)
        user_input = qa_pair.get('user_input', '')[:30]
        logger.info(f"STM: Added QA. User: {user_input}...")
        self.save()

    def get_all(self) -> List[Dict[str, Any]]:
        """
        获取所有短期记忆
        
        Returns:
            短期记忆列表
        """
        return list(self.memory)

    def is_full(self) -> bool:
        """
        检查短期记忆是否已满
        
        Returns:
            是否已满
        """
        # 当短期记忆满时，需要迁移到中期记忆
        return len(self.memory) >= self.max_capacity

    def pop_oldest(self) -> Optional[Dict[str, Any]]:
        """
        弹出最旧的记忆
        
        Returns:
            最旧的记忆，如果没有则返回 None
        """
        if self.memory:
            msg = self.memory.popleft()
            logger.info('STM: Evicted oldest QA pair.')
            self.save()
            return msg
        return None

    def save(self):
        """
        保存短期记忆到文件
        """
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.memory), f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Error saving STM to {self.file_path}: {e}")

    def load(self):
        """
        从文件加载短期记忆
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.memory = deque(data, maxlen=self.max_capacity)
                else:
                    self.memory = deque(maxlen=self.max_capacity)
            logger.info(f"STM: Loaded from {self.file_path}.")
        except FileNotFoundError:
            self.memory = deque(maxlen=self.max_capacity)
            logger.info(f"STM: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            self.memory = deque(maxlen=self.max_capacity)
            logger.error(f"STM: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            self.memory = deque(maxlen=self.max_capacity)
            logger.error(f"STM: Unexpected error loading from {self.file_path}: {e}. Initializing new memory.")