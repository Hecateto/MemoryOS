import json
import logging
import numpy as np
from collections import defaultdict
import faiss
import heapq
from typing import List, Dict, Any, Optional, Tuple

try:
    from .utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector,
        compute_time_decay, ensure_directory_exists, OpenAIClient
    )
except ImportError:
    from utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector,
        compute_time_decay, ensure_directory_exists, OpenAIClient
    )

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Heat parameters
HEAT_ALPHA = 1.0        # 访问次数权重
HEAT_BETA = 1.0         # 交互强度权重
HEAT_GAMMA = 1.0        # 最近访问时间衰减权重
RECENCY_TAU_HOURS = 24   # 时间衰减常数（小时）

def compute_segment_heat(session: Dict[str, Any],
                         alpha: float = HEAT_ALPHA, beta: float = HEAT_BETA, 
                         gamma: float = HEAT_GAMMA, tau_hours: float = RECENCY_TAU_HOURS) -> float:
    """
    计算会话热度
    H_segment = α × N_visit + β × L_interaction + γ × R_recency

    N_visit：访问次数
    L_interaction：交互强度（如对话轮数等）
    R_recency：最近访问时间衰减（使用指数衰减函数计算，τ为衰减时间常数）
    R = exp(-Δt / τ)，其中Δt是当前时间与上次访问时间的差值，τ是一个可调节的时间常数，控制衰减速度。
        R_recency
      1.0 ┼────╮
           │     ╲
      0.5 ┼       ╲
           │         ╲___
      0.0 ┼──────────────────▶ time
           0   24h   48h   72h

    Args:
        session: 会话对象
        alpha: 访问次数权重
        beta: 交互强度权重
        gamma: 时间衰减权重
        tau_hours: 时间衰减常数
    
    Returns:
        热度值
    """
    N_visit = session.get('N_visit', 0)
    L_interaction = session.get('L_interaction', 0)

    R_recency = 1.0
    if session.get('last_visit_time'):
        R_recency = compute_time_decay(session['last_visit_time'], get_timestamp(), tau_hours)
    session['R_recency'] = R_recency
    return alpha * N_visit + beta * L_interaction + gamma * R_recency

"""
session_obj = {
    "id": session_id,                    # 会话唯一标识
    "summary": summary,                  # 会话主题摘要（用于相似度匹配）
    "summary_keywords": summary_keywords,# 主题关键词（辅助匹配）
    "summary_embedding": summary_vec,    # 摘要的向量表示（用于 FAISS 检索）
    
    "details": processed_details,        # 包含的 Pages 列表（具体对话内容）
    
    # ↓↓↓ 核心：热度计算因子 ↓↓↓
    "L_interaction": len(processed_details),  # 交互长度（对话轮数）
    "R_recency": 1.0,                         # 时间衰减因子
    "N_visit": 0,                             # 访问次数
    "H_segment": 0.0,                         # 热度值（由上述因子计算）
    
    "timestamp": current_ts,             # 创建时间
    "last_visit_time": current_ts,       # 最后访问时间
    "access_count_lfu": 0                # LFU 淘汰用的访问计数
}
"""

class MidTermMemory:
    """
    中期记忆类，用于存储和管理会话级别的记忆
    使用向量检索和热度计算来优化记忆管理
    """
    def __init__(self, file_path: str, client: OpenAIClient, max_capacity: int = 2000,
                 embedding_model_name: str = "all-MiniLM-L6-v2", embedding_model_kwargs: dict = None,
                 use_embedding_api: bool = False):
        """
        初始化中期记忆
        
        Args:
            file_path: 存储文件路径
            client: OpenAIClient 实例
            max_capacity: 最大容量
            embedding_model_name: 嵌入模型名称
            embedding_model_kwargs: 嵌入模型参数
            use_embedding_api: 是否使用 API 获取嵌入
        """
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        self.client = client
        self.max_capacity = max_capacity
        self.sessions = {}  # {sid: session_obj}
        self.access_frequency = defaultdict(int) # {sid: count}
        self.heap = [] # Min-heap for eviction based on heat (-H_segment, sid)

        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs or {}
        self.use_embedding_api = use_embedding_api
        self.load()

    def get_page_by_id(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        根据页面 ID 获取页面
        
        Args:
            page_id: 页面 ID
        
        Returns:
            页面对象，如果不存在则返回 None
        """
        for session in self.sessions.values():
            for page in session.get('details', []):
                if page.get('page_id') == page_id:
                    return page
        return None

    def update_page_connections(self, prev_page_id: Optional[str], next_page_id: Optional[str]):
        """
        更新页面连接关系
        
        Args:
            prev_page_id: 前一页 ID
            next_page_id: 后一页 ID
        """
        if prev_page_id:
            prev_page = self.get_page_by_id(prev_page_id)
            if prev_page:
                prev_page['next_page'] = next_page_id
        if next_page_id:
            next_page = self.get_page_by_id(next_page_id)
            if next_page:
                next_page['prev_page'] = prev_page_id

    def evict_lfu(self):
        """
        使用 LFU 策略淘汰会话
        """
        if not self.access_frequency or not self.sessions:
            return
        
        lfu_sid = min(self.access_frequency, key=self.access_frequency.get)
        logger.info(f"MTM: Evicting session {lfu_sid} with access count {self.access_frequency[lfu_sid]}")

        if lfu_sid not in self.sessions:
            del self.access_frequency[lfu_sid]
            self.rebuild_heap()
            return

        session_to_delete = self.sessions.pop(lfu_sid)
        del self.access_frequency[lfu_sid]

        # 清理页面连接
        for page in session_to_delete.get('details', []):
            prev_page_id = page.get('prev_page')
            next_page_id = page.get('next_page')
            if prev_page_id and not self.get_page_by_id(prev_page_id):
                pass
            if next_page_id and not self.get_page_by_id(next_page_id):
                pass

        self.rebuild_heap()
        self.save()
        logger.info(f"MTM: Evicted session {lfu_sid}. Remaining sessions: {len(self.sessions)}")

    def add_session(self, summary: str, details: List[Dict[str, Any]], 
                    summary_keywords: Optional[List[str]] = None) -> str:
        """
        添加新会话
        
        Args:
            summary: 会话摘要
            details: 页面列表
            summary_keywords: 会话关键词
        
        Returns:
            会话 ID
        """
        session_id = generate_id('session')
        
        # 生成摘要嵌入
        summary_vec = get_embedding(
            summary,
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        summary_vec = normalize_vector(summary_vec).tolist()
        summary_keywords = summary_keywords or []

        # 处理页面
        processed_details = []
        for page_data in details:
            page_id = page_data.get('page_id', generate_id('page'))
            
            # 处理页面嵌入
            if 'page_embedding' in page_data and page_data['page_embedding']:
                logger.debug(f"MTM: Reusing existing embedding for page {page_id}")
                inp_vec = page_data['page_embedding']
                if isinstance(inp_vec, list):
                    inp_vec = np.array(inp_vec, dtype=np.float32)
                    if np.linalg.norm(inp_vec) > 1.1 or np.linalg.norm(inp_vec) < 0.9:
                        inp_vec = normalize_vector(inp_vec).tolist()
            else:
                logger.debug(f"MTM: Generating new embedding for page {page_id}")
                full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                inp_vec = get_embedding(
                    full_text,
                    model_name=self.embedding_model_name,
                    use_api=self.use_embedding_api,
                    **self.embedding_model_kwargs
                )
                inp_vec = normalize_vector(inp_vec).tolist()

            # 处理页面关键词
            if 'page_keywords' in page_data and page_data['page_keywords']:
                logger.debug(f"MTM: Reusing existing keywords for page {page_id}")
                page_keywords = page_data['page_keywords']
            else:
                logger.debug(f"MTM: Setting empty keywords for page {page_id}, which will be generated by multi-summary")
                page_keywords = []

            processed_page = {
                **page_data,
                'page_id': page_id,
                'page_embedding': inp_vec,
                'page_keywords': page_keywords,
                'preloaded': page_data.get('preloaded', False),
                'analyzed': page_data.get('analyzed', False)
            }
            processed_details.append(processed_page)

        # 创建会话对象
        session_obj = {
            "id": session_id,
            "summary": summary,
            "summary_keywords": summary_keywords,
            "summary_embedding": summary_vec,
            "details": processed_details,
            "L_interaction": len(processed_details),
            "R_recency": 1.0,
            "N_visit": 0,
            "H_segment": 0.0,
            "timestamp": get_timestamp(),
            "last_visit_time": get_timestamp(),
            "access_count_lfu": 0
        }
        
        # 计算热度
        session_obj['H_segment'] = compute_segment_heat(session_obj)
        
        # 添加到会话字典和堆中
        self.sessions[session_id] = session_obj
        self.access_frequency[session_id] = 0
        heapq.heappush(self.heap, (-session_obj['H_segment'], session_id))

        logger.info(f"MTM: Added session {session_id}, Total sessions: {len(self.sessions)}")
        
        # 检查容量并淘汰
        if len(self.sessions) > self.max_capacity:
            self.evict_lfu()
        
        # 保存
        self.save()
        return session_id

    def rebuild_heap(self):
        """
        重建堆
        """
        self.heap = []
        for sid, session_data in self.sessions.items():
            heapq.heappush(self.heap, (-session_data['H_segment'], sid))

    def insert_pages_into_session(self, summary_for_new_pages: str, 
                                 keywords_for_new_pages: List[str],
                                 pages_to_insert: List[Dict[str, Any]],
                                 similarity_threshold: float = 0.5,
                                 keyword_similarity_alpha: float = 1.0) -> str:
        """
        将新页面插入到最匹配的会话中，若无合适会话则新建会话
        
        Args:
            summary_for_new_pages: 新页面的摘要
            keywords_for_new_pages: 新页面的关键词
            pages_to_insert: 要插入的页面列表
            similarity_threshold: 相似度阈值
            keyword_similarity_alpha: 关键词相似度权重
        
        Returns:
            会话 ID
        """
        if not self.sessions:
            logger.info("MTM: No existing sessions. Adding new session.")
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

        # 生成新摘要的嵌入
        new_summary_vec = get_embedding(
            summary_for_new_pages,
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        new_summary_vec = normalize_vector(new_summary_vec)

        # 寻找最佳匹配的会话
        best_sid = None
        best_overall_score = -float('inf')
        
        for sid, session in self.sessions.items():
            # 摘要的语义相似度
            summary_vec = np.array(session['summary_embedding'], dtype=np.float32)
            semantic_sim = float(np.dot(summary_vec, new_summary_vec))
            
            # 关键词的Jaccard相似度
            # existing_keywords = set(session.get('summary_keywords', []))
            # new_keywords = set(keywords_for_new_pages)
            # s_topic_keywords = 0
            # if existing_keywords and new_keywords:
            #     intersection = len(existing_keywords.intersection(new_keywords))
            #     union = len(existing_keywords.union(new_keywords))
            #     if union > 0:
            #         s_topic_keywords = intersection / union

            # 关键词的包含匹配（Substring Matching）相似度
            existing_keywords_list = set(session.get('summary_keywords', []))
            new_keywords_list = set(keywords_for_new_pages)
            s_topic_keywords = 0

            if existing_keywords_list and new_keywords_list:
                for nk in new_keywords_list:
                    for ek in existing_keywords_list:
                        # 互为子串
                        if nk.lower() in ek.lower() or ek.lower() in nk.lower():
                            s_topic_keywords += 0.4
                            break
            
            # 综合相似度得分 = 语义相似度 + α * 关键词相似度
            overall_score = semantic_sim + keyword_similarity_alpha * s_topic_keywords
            logger.info(f"MTM: Session {sid} - Semantic Sim: {semantic_sim:.4f}, Keyword Sim: {s_topic_keywords:.4f}, Overall Score: {overall_score:.4f}")
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_sid = sid

        logger.info(f"MTM: Best matching session {best_sid} with score {best_overall_score:.4f}")
        # 合并到最佳匹配的会话
        if best_sid and best_overall_score >= similarity_threshold:
            logger.info(f"MTM: Inserting pages into existing session {best_sid} with score {best_overall_score:.4f}")
            target_session = self.sessions[best_sid]
            
            for page_data in pages_to_insert:
                page_id = page_data.get('page_id', generate_id('page'))
                
                # 处理页面嵌入
                if 'page_embedding' in page_data and page_data['page_embedding']:
                    logger.debug(f"MTM: Reusing existing embedding for new page {page_id}")
                    inp_vec = page_data['page_embedding']
                    if isinstance(inp_vec, list):
                        inp_vec = np.array(inp_vec, dtype=np.float32)
                        if np.linalg.norm(inp_vec) > 1.1 or np.linalg.norm(inp_vec) < 0.9:
                            inp_vec = normalize_vector(inp_vec).tolist()
                else:
                    logger.debug(f"MTM: Generating new embedding for new page {page_id}")
                    full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                    inp_vec = get_embedding(
                        full_text,
                        model_name=self.embedding_model_name,
                        use_api=self.use_embedding_api,
                        **self.embedding_model_kwargs
                    )
                    inp_vec = normalize_vector(inp_vec).tolist()

                # 处理页面关键词
                if 'page_keywords' in page_data and page_data['page_keywords']:
                    logger.debug(f"MTM: Reusing existing keywords for new page {page_id}")
                    page_keywords = page_data['page_keywords']
                else:
                    logger.debug(f"MTM: Setting session keywords for new page {page_id}")
                    page_keywords = keywords_for_new_pages

                processed_page = {
                    **page_data,
                    'page_id': page_id,
                    'page_embedding': inp_vec,
                    'page_keywords': page_keywords,
                }
                target_session['details'].append(processed_page)

            # 更新会话信息
            target_session['L_interaction'] = len(target_session['details'])
            target_session['last_visit_time'] = get_timestamp()
            target_session['H_segment'] = compute_segment_heat(target_session)

            # 更新（合并）关键词
            merged_keywords = set(target_session.get('summary_keywords', []))
            merged_keywords.update(keywords_for_new_pages)
            target_session['summary_keywords'] = list(merged_keywords)

            """
            - Threshold
                - 这里给予了keywords重复极大的权重
                - 也可以考虑降低semantic_threshold
            - Update
                - 这里仅采用了更新keywords
                - 也可以考虑更新summary和对应emb, 但会频繁调用emb_model
                - 解决办法是记录session更新次数, 超过阈值触发summary重写 (控制在一定字数内)
            """

            # 重建堆并保存
            self.rebuild_heap()
            self.save()
            return best_sid
        # 否则新建会话
        else:
            logger.info("MTM: No suitable session found. Adding new session.")
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

    def search_sessions(self, query_text: str,
                        segment_similarity_threshold: float = 0.4,
                        page_similarity_threshold: float = 0.4,
                        top_k_sessions: int = 5, keyword_alpha: float = 1.0, 
                        recency_tau_search: int = 3600) -> List[Dict[str, Any]]:
        """
        搜索与查询文本最相关的会话和页面

        Retrieve -> Segments (Top-k, Similarity Threshold) -> Pages (Similarity Threshold)
        Sim_score: semantic + [α * Jaccard]
        
        Args:
            query_text: 查询文本
            segment_similarity_threshold: 会话相似度阈值
            page_similarity_threshold: 页面相似度阈值
            top_k_sessions: 返回的会话数量
            keyword_alpha: 关键词相似度权重
            recency_tau_search: 时间衰减常数
        
        Returns:
            搜索结果列表
        """
        if not self.sessions:
            return []
        
        # 生成查询嵌入
        query_vec = get_embedding(
            query_text,
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec)
        query_vec_matrix = query_vec.reshape(1, -1)

        # 准备会话数据
        session_ids = []
        summary_embeddings = []
        for sid, data in self.sessions.items():
            if 'summary_embedding' in data and data['summary_embedding']:
                session_ids.append(sid)
                summary_embeddings.append(data['summary_embedding'])

        if not session_ids:
            return []

        # 使用 FAISS 进行相似度搜索
        summary_mat = np.array(summary_embeddings, dtype=np.float32)
        dim = summary_mat.shape[1]

        # 余弦相似度搜索
        index = faiss.IndexFlatIP(dim)
        index.add(summary_mat)
        distances, indices = index.search(query_vec_matrix, min(top_k_sessions, len(session_ids)))

        results = []
        cur_time_str = get_timestamp()

        # 页面级过滤
        for idx, session_relevance_score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            
            sid = session_ids[idx]
            session = self.sessions[sid]

            if session_relevance_score < segment_similarity_threshold:
                continue

            pages = session.get('details', [])
            if not pages:
                continue

            # 计算页面相似度
            page_embeddings = [p['page_embedding'] for p in pages]
            page_mat = np.array(page_embeddings, dtype=np.float32)
            page_sim_scores = np.dot(page_mat, query_vec)

            # 过滤相似度高于阈值的页面
            mask = page_sim_scores >= page_similarity_threshold
            if not np.any(mask):
                continue

            valid_indices = np.where(mask)[0]
            matched_pages_in_session = []

            for p_idx in valid_indices:
                matched_pages_in_session.append({
                    'page_data': pages[p_idx],
                    'score': float(page_sim_scores[p_idx])
                })

            if matched_pages_in_session:
                # 更新会话访问信息
                session['N_visit'] += 1
                session['last_visit_time'] = cur_time_str
                session['access_count_lfu'] = session.get('access_count_lfu', 0) + 1
                if hasattr(self, 'access_frequency'):
                    self.access_frequency[sid] += 1

                # 更新热度
                session['H_segment'] = compute_segment_heat(session)
                self.rebuild_heap()

                # 排序并添加结果
                matched_pages_in_session.sort(key=lambda x: x['score'], reverse=True)
                results.append({
                    'session_id': sid,
                    'session_summary': session['summary'],
                    'session_relevance_score': session_relevance_score,
                    'matched_pages': matched_pages_in_session
                })

        # 保存并返回排序后的结果
        self.save()
        return sorted(results, key=lambda x: x['session_relevance_score'], reverse=True)

    def save(self):
        """
        保存中期记忆到文件
        """
        data_to_save = {
            'sessions': self.sessions,
            'access_frequency': dict(self.access_frequency),
        }
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving mid-term memory: {e}")

    def load(self):
        """
        从文件加载中期记忆
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.sessions = data.get('sessions', {})
                self.access_frequency = defaultdict(int, data.get('access_frequency', {}))
                self.rebuild_heap()
            logger.info(f"MTM: Loaded from {self.file_path}, Sessions: {len(self.sessions)}")
        except FileNotFoundError:
            logger.info(f"MTM: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            logger.error(f"MTM: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            logger.error(f"MTM: Error loading mid-term memory: {e}. Initializing new memory.")

