import logging
from collections import deque
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

try:
    from .utils import get_timestamp, OpenAIClient, run_parallel_tasks
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory
    from .long_term import LongTermMemory
except ImportError:
    from utils import get_timestamp, OpenAIClient, run_parallel_tasks
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory
    from long_term import LongTermMemory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Retriever:
    """
    记忆检索器类，负责从中期记忆和长期记忆中检索相关上下文
    """
    def __init__(self,
                 mid_term_memory: MidTermMemory,
                 long_term_memory: LongTermMemory,
                 assistant_long_term_memory: Optional[LongTermMemory] = None,
                 queue_capacity: int = 7):
        """
        初始化检索器
        
        Args:
            mid_term_memory: 中期记忆实例
            long_term_memory: 长期记忆实例
            assistant_long_term_memory: 助手长期记忆实例
            queue_capacity: 检索队列容量
        """
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.assistant_long_term_memory = assistant_long_term_memory  # Store assistant LTM reference
        self.retrieval_queue_capacity = queue_capacity

    def _retrieve_mid_term_context(self, user_query: str,
                                   segment_similarity_threshold: float,
                                   page_similarity_threshold: float,
                                   top_k_sessions: int) -> List[Dict[str, Any]]:
        """
        检索中期记忆上下文
        
        Args:
            user_query: 用户查询
            segment_similarity_threshold: 会话相似度阈值
            page_similarity_threshold: 页面相似度阈值
            top_k_sessions: 返回的会话数量
            
        Returns:
            检索到的页面列表
        """
        logger.info("Retriever: Searching MTM...")
        matched_sessions = self.mid_term_memory.search_sessions(
            query_text=user_query,
            segment_similarity_threshold=segment_similarity_threshold,
            page_similarity_threshold=page_similarity_threshold,
            top_k_sessions=top_k_sessions
        )
        top_pages_heap = []
        page_counter = 0
        for session in matched_sessions:
            for page in session.get("matched_pages", []):
                page_data = page["page_data"]
                page_score = page["score"]
                combined_score = page_score

                if len(top_pages_heap) < self.retrieval_queue_capacity:
                    heapq.heappush(top_pages_heap, (combined_score, page_counter, page_data))
                elif combined_score > top_pages_heap[0][0]:
                    heapq.heappop(top_pages_heap)
                    heapq.heappush(top_pages_heap, (combined_score, page_counter, page_data))
                page_counter += 1

        retrieved_pages = [item[2] for item in sorted(top_pages_heap, key=lambda x: x[0], reverse=True)]
        return retrieved_pages

    def _retrieve_user_knowledge(self, user_query: str, knowledge_threshold: float, top_k_knowledge: int) -> List[Dict[str, Any]]:
        """
        检索用户知识
        
        Args:
            user_query: 用户查询
            knowledge_threshold: 知识相似度阈值
            top_k_knowledge: 返回的知识数量
            
        Returns:
            检索到的用户知识列表
        """
        logger.info("Retriever: Searching user LTM...")
        retrieved_knowledge = self.long_term_memory.search_user_knowledge(
            query=user_query, threshold=knowledge_threshold, top_k=top_k_knowledge
        )
        return retrieved_knowledge

    def _retrieve_assistant_knowledge(self, user_query: str, knowledge_threshold: float, top_k_knowledge: int) -> List[Dict[str, Any]]:
        """
        检索助手知识
        
        Args:
            user_query: 用户查询
            knowledge_threshold: 知识相似度阈值
            top_k_knowledge: 返回的知识数量
            
        Returns:
            检索到的助手知识列表
        """
        if not self.assistant_long_term_memory:
            return []
        logger.info("Retriever: Searching assistant LTM...")
        retrieved_knowledge = self.assistant_long_term_memory.search_assistant_knowledge(
            query=user_query, threshold=knowledge_threshold, top_k=top_k_knowledge
        )
        return retrieved_knowledge

    def retrieve_context(self, user_query: str,
                         user_id: str,
                         segment_similarity_threshold: float = 0.4,
                         page_similarity_threshold: float = 0.4,
                         knowledge_threshold: float = 0.1,
                         top_k_sessions: int = 5,
                         top_k_knowledge: int = 20
                         ) -> Dict[str, Any]:
        """
        并行从中期记忆和长期知识库中检索相关上下文
        - 中期记忆检索关注最近的交互和相关会话片段
        - 长期记忆检索关注用户特定知识和助手的知识库
        
        Args:
            user_query: 用户查询
            user_id: 用户 ID
            segment_similarity_threshold: 会话相似度阈值
            page_similarity_threshold: 页面相似度阈值
            knowledge_threshold: 知识相似度阈值
            top_k_sessions: 返回的会话数量
            top_k_knowledge: 返回的知识数量
            
        Returns:
            检索结果，包含检索到的页面、用户知识和助手知识
        """
        tasks = [
            lambda: self._retrieve_mid_term_context(user_query, segment_similarity_threshold, page_similarity_threshold, top_k_sessions),
            lambda: self._retrieve_user_knowledge(user_query, knowledge_threshold, top_k_knowledge),
            lambda: self._retrieve_assistant_knowledge(user_query, knowledge_threshold, top_k_knowledge)
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, task in enumerate(tasks):
                future = executor.submit(task)
                futures.append((i, future))
            results = [None] * 3
            for task_idx, future in futures:
                try:
                    results[task_idx] = future.result()
                except Exception as e:
                    logger.error(f"Retriever: Error in task {task_idx} - {str(e)}")
                    results[task_idx] = []

        retrieved_mid_term_pages, retrieved_user_knowledge, retrieved_assistant_knowledge = results

        return {
            "retrieved_pages": retrieved_mid_term_pages or [],
            "retrieved_user_knowledge": retrieved_user_knowledge or [],
            "retrieved_assistant_knowledge": retrieved_assistant_knowledge or [],
            "retrieved_at": get_timestamp()
        }