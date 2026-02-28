import logging
from typing import List, Dict, Any, Optional

try:
    from .utils import (
        generate_id, get_timestamp,
        gpt_generate_multi_summary, check_conversation_continuity, generate_page_meta_info, OpenAIClient,
        run_parallel_tasks
    )
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory
    from .long_term import LongTermMemory
except ImportError:
    from utils import (
        generate_id, get_timestamp,
        gpt_generate_multi_summary, check_conversation_continuity, generate_page_meta_info, OpenAIClient,
        run_parallel_tasks
    )
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory
    from long_term import LongTermMemory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Updater:
    """
    记忆更新器类，负责将短期记忆更新到中期记忆，并从中期记忆更新到长期记忆
    """
    def __init__(self,
                 short_term_memory: ShortTermMemory,
                 mid_term_memory: MidTermMemory,
                 long_term_memory: LongTermMemory,
                 client: OpenAIClient, 
                 llm_model: str, 
                 topic_similarity_threshold: float = 0.5):
        """
        初始化更新器
        
        Args:
            short_term_memory: 短期记忆实例
            mid_term_memory: 中期记忆实例
            long_term_memory: 长期记忆实例
            client: OpenAI 客户端
            llm_model: LLM 模型名称
            topic_similarity_threshold: 主题相似度阈值
        """
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.client = client
        self.llm_model = llm_model
        self.topic_similarity_threshold = topic_similarity_threshold
        self.last_evicted_page_for_continuity = None

    def _process_page_embedding_and_keywords(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理页面嵌入和关键词
        
        Args:
            page_data: 页面数据
            
        Returns:
            处理后的页面数据
        """
        page_id = page_data.get('page_id', generate_id())

        if "page_embedding" in page_data and page_data["page_embedding"]:
            return page_data

        if not ("page_embedding" in page_data and page_data["page_embedding"]):
            full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
            try:
                emb = self._get_embedding_for_page(full_text)
                if emb is not None:
                    from .utils import normalize_vector
                    page_data["page_embedding"] = normalize_vector(emb).tolist()
            except Exception as e:
                logger.error(f"Error generating embedding for page {page_id}: {e}")

        if "page_keywords" not in page_data:
            page_data["page_keywords"] = []
        return page_data

    def _get_embedding_for_page(self, text: str) -> Any:
        """
        获取页面文本的嵌入
        
        Args:
            text: 页面文本
            
        Returns:
            嵌入向量
        """
        from .utils import get_embedding
        return get_embedding(
            text, model_name=self.mid_term_memory.embedding_model_name,
            use_api=self.mid_term_memory.use_embedding_api,
            **self.mid_term_memory.embedding_model_kwargs
        )

    def _update_linked_pages_meta_info(self, start_page_id: str, new_meta_info: str):
        """
        更新链接页面的元信息
        
        Args:
            start_page_id: 起始页面 ID
            new_meta_info: 新的元信息
        """
        q = [start_page_id]
        vis = {start_page_id}

        head = 0
        while head < len(q):
            cur_page_id = q[head]
            head += 1
            page = self.mid_term_memory.get_page_by_id(cur_page_id)
            if page is None:
                continue
            page["meta_info"] = new_meta_info
            logger.info(f"Updated meta_info for page {cur_page_id} to: {new_meta_info}")
            prev_id = page.get("prev_page")
            if prev_id and prev_id not in vis:
                vis.add(prev_id)
                q.append(prev_id)
            next_id = page.get("next_page")
            if next_id and next_id not in vis:
                vis.add(next_id)
                q.append(next_id)
        if q:
            self.mid_term_memory.save()

    def process_short_term_to_mid_term(self):
        """
        将短期记忆处理到中期记忆
        STM -> MTM
        """
        # 考虑add_memory的原子性, 在STM队列首次满之后, 每次add_memory只弹出一个qa对转换为MTM
        # 可修改为弹出一个batch的qa对进行处理, 以减少频次
        # if self.short_term_memory.is_full():
        # batch_size = max(1, self.short_term_memory.max_capacity // 2)
        #     for _ in range(batch_size):
        #         if len(self.short_term_memory.memory) == 0:
        #             break
        #         qa = self.short_term_memory.pop_oldest()
        #         if qa and qa.get("user_input") and qa.get("agent_response"):
        #             evicted_qa.append(qa)

        evicted_qa = []
        while self.short_term_memory.is_full():
            qa = self.short_term_memory.pop_oldest()
            if qa and qa.get("user_input") and qa.get("agent_response"):
                evicted_qa.append(qa)

        if not evicted_qa:
            return

        logger.info(f"Updater: Processing {len(evicted_qa)} QAs from short-term to mid-term.")

        cur_batch_pages = []
        temp_last_page_in_batch = self.last_evicted_page_for_continuity

        for qa in evicted_qa:
            cur_page_obj = {
                "page_id": generate_id("page"),
                "user_input": qa.get("user_input", ""),
                "agent_response": qa.get("agent_response", ""),
                "timestamp": qa.get("timestamp", get_timestamp()),
                "preloaded": False,
                "analyzed": False,
                "prev_page": None,
                "next_page": None,
                "meta_info": None,
            }

            is_continuous = check_conversation_continuity(temp_last_page_in_batch, cur_page_obj,
                                                          self.client, model=self.llm_model)
            if is_continuous and temp_last_page_in_batch:
                cur_page_obj["prev_page"] = temp_last_page_in_batch["page_id"]
                # temp_last_page_in_batch["next_page"] = cur_page_obj["page_id"]
                last_meta = temp_last_page_in_batch.get("meta_info")
                new_meta = generate_page_meta_info(last_meta, cur_page_obj, self.client, model=self.llm_model)
                cur_page_obj["meta_info"] = new_meta
                if temp_last_page_in_batch.get("page_id") and \
                        self.mid_term_memory.get_page_by_id(temp_last_page_in_batch["page_id"]):
                    self._update_linked_pages_meta_info(temp_last_page_in_batch["page_id"], new_meta)
                logger.info(f"Page {cur_page_obj['page_id']} is continuous with previous page {temp_last_page_in_batch['page_id']}. Updated meta_info to: {new_meta}")
            else:
                cur_page_obj["meta_info"] = generate_page_meta_info(None, cur_page_obj, self.client, model=self.llm_model)
                logger.info(f"Page {cur_page_obj['page_id']} is NOT continuous with previous page. Generated meta_info: {cur_page_obj['meta_info']}")

            cur_batch_pages.append(cur_page_obj)
            # 这里一视同仁的将当前页面设置为下一轮的上一页面，可能忽略了连续性断开的情况
            temp_last_page_in_batch = cur_page_obj

        if cur_batch_pages:
            self.last_evicted_page_for_continuity = cur_batch_pages[-1]

        if not cur_batch_pages:
            return

        input_text_for_summary = "\n".join([
            f"User: {p.get('user_input','')}\nAssistant: {p.get('agent_response','')}"
            for p in cur_batch_pages
        ])

        logger.info(f"Input text for multi-topic summary:\n{input_text_for_summary}")

        logger.info("Updater: Generating multi-topic summary for the evicted batch...")
        multi_summary_result = gpt_generate_multi_summary(input_text_for_summary, self.client, model=self.llm_model)

        # {"summaries": [{"theme": "...", "keywords": [...], "content": "..."}]}
        if multi_summary_result and multi_summary_result.get("summaries"):
            for summary_item in multi_summary_result["summaries"]:
                theme_summary = summary_item.get("content", "General summary of recent interactions.")
                theme_keywords = summary_item.get("keywords", [])
                self.mid_term_memory.insert_pages_into_session(
                    summary_for_new_pages=theme_summary,
                    keywords_for_new_pages=theme_keywords,
                    pages_to_insert=cur_batch_pages,
                    similarity_threshold=self.topic_similarity_threshold
                )
        else:
            fallback_summary = "General conversation segment from short-term memory."
            fallback_keywords = []
            self.mid_term_memory.insert_pages_into_session(
                summary_for_new_pages=fallback_summary,
                keywords_for_new_pages=fallback_keywords,
                pages_to_insert=cur_batch_pages,
                similarity_threshold=self.topic_similarity_threshold
            )

        for page in cur_batch_pages:
            if page.get("prev_page"):
                self.mid_term_memory.update_page_connections(page["prev_page"], page["page_id"])
            if page.get("next_page"):
                self.mid_term_memory.update_page_connections(page["page_id"], page["next_page"])
        if cur_batch_pages:
            self.mid_term_memory.save()

    def update_long_term_from_analysis(self, user_id: str, profile_analysis_result: Dict[str, Any]):
        """
        根据画像分析结果更新长期记忆
        
        Args:
            user_id: 用户 ID
            profile_analysis_result: 画像分析结果
        """
        if not profile_analysis_result:
            return

        logger.info("Updater: Updating long-term memory based on profile analysis results...")

        new_profile = profile_analysis_result.get("profile")
        user_private_knowledge = profile_analysis_result.get("private")
        assistant_knowledge = profile_analysis_result.get("assistant_knowledge")

        if new_profile and new_profile.lower() != "none":
            self.long_term_memory.update_user_profile(user_id, new_profile, merge=False)

        if user_private_knowledge and user_private_knowledge.lower() != "none":
            for line in user_private_knowledge.split("\n"):
                if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                    self.long_term_memory.add_user_knowledge(line.strip())

        if assistant_knowledge and assistant_knowledge.lower() != "none":
            for line in assistant_knowledge.split("\n"):
                if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                    self.long_term_memory.add_assistant_knowledge(line.strip())