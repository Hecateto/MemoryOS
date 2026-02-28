import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 修改为绝对导入
try:
    # 尝试相对导入（当作为包使用时）
    from .utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, ensure_directory_exists, set_embedding_api_client
    from . import prompts
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory, compute_segment_heat # For H_THRESHOLD logic
    from .long_term import LongTermMemory
    from .updater import Updater
    from .retriever import Retriever
except ImportError:
    # 回退到绝对导入（当作为独立模块使用时）
    from utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, ensure_directory_exists, set_embedding_api_client
    import prompts
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory, compute_segment_heat # For H_THRESHOLD logic
    from long_term import LongTermMemory
    from updater import Updater
    from retriever import Retriever

# 中期记忆热度阈值（超过则更新长期画像）
H_PROFILE_UPDATE_THRESHOLD = 5.0
DEFAULT_ASSISTANT_ID = "default_assistant_profile"

class MemoryOS:
    """
    MemoryOS 核心类，管理用户的短期、中期和长期记忆
    提供记忆存储、检索和响应生成功能
    """
    def __init__(self, user_id: str,
                 openai_api_key: str,
                 data_storage_path: str,
                 openai_base_url: str = None,
                 assistant_id: str = DEFAULT_ASSISTANT_ID,
                 short_term_capacity: int = 10,
                 mid_term_capacity: int = 2000,
                 long_term_knowledge_capacity: int = 100,
                 retrieval_queue_capacity: int = 7,
                 mid_term_heat_threshold: float = H_PROFILE_UPDATE_THRESHOLD,
                 mid_term_similarity_threshold: float = 0.5,
                 llm_model: str = "gpt-4o-mini",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 embedding_model_kwargs: dict = None,
                 use_embedding_api: bool = False,  # 新增：是否使用API调用embedding服务
                 ):
        """
        初始化 MemoryOS 实例
        
        Args:
            user_id: 用户 ID
            openai_api_key: OpenAI API 密钥
            data_storage_path: 数据存储路径
            openai_base_url: OpenAI API 基础 URL
            assistant_id: 助手 ID
            short_term_capacity: 短期记忆容量
            mid_term_capacity: 中期记忆容量
            long_term_knowledge_capacity: 长期记忆知识容量
            retrieval_queue_capacity: 检索队列容量
            mid_term_heat_threshold: 中期记忆热度阈值
            mid_term_similarity_threshold: 中期记忆相似度阈值
            llm_model: LLM 模型名称
            embedding_model_name: 嵌入模型名称
            embedding_model_kwargs: 嵌入模型参数
            use_embedding_api: 是否使用 API 调用嵌入服务
        """
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = data_storage_path
        self.llm_model = llm_model
        self.mid_term_similarity_threshold = mid_term_similarity_threshold
        self.mid_term_heat_threshold = mid_term_heat_threshold
        self.embedding_model_name = embedding_model_name
        self.use_embedding_api = use_embedding_api

        if embedding_model_kwargs is None:
            if not use_embedding_api and 'bge-m3' in self.embedding_model_name.lower():
                logger.info("Detected bge-m3 model, defaulting embedding_model_kwargs to {'use_fp16': True}")
                self.embedding_model_kwargs = {'use_fp16': True}
            else:
                self.embedding_model_kwargs = {}
        else:
            self.embedding_model_kwargs = embedding_model_kwargs

        logger.info(f"Initializing MemoryOS for user '{self.user_id}' and assistant '{self.assistant_id}'. Data path: {self.data_storage_path}")
        logger.info(f"Using unified LLM model: {self.llm_model}")
        logger.info(f"Using embedding model: {self.embedding_model_name} (use_api={self.use_embedding_api}) with kwargs: {self.embedding_model_kwargs}")

        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)
        if self.use_embedding_api:
            set_embedding_api_client(self.client)

        # 定义用户特定数据的文件路径
        self.user_data_dir = os.path.join(self.data_storage_path, "users", self.user_id)
        user_short_term_path = os.path.join(self.user_data_dir, "short_term.json")
        user_mid_term_path = os.path.join(self.user_data_dir, "mid_term.json")
        user_long_term_path = os.path.join(self.user_data_dir,
                                           "long_term_user.json")  # User profile and their knowledge

        # 定义助手特定数据的文件路径
        self.assistant_data_dir = os.path.join(self.data_storage_path, "assistants", self.assistant_id)
        assistant_long_term_path = os.path.join(self.assistant_data_dir, "long_term_assistant.json")

        # 确保目录存在
        ensure_directory_exists(user_short_term_path)
        ensure_directory_exists(user_mid_term_path)
        ensure_directory_exists(user_long_term_path)
        ensure_directory_exists(assistant_long_term_path)

        # 初始化用户记忆模块
        self.short_term_memory = ShortTermMemory(file_path=user_short_term_path, max_capacity=short_term_capacity)
        self.mid_term_memory = MidTermMemory(
            file_path=user_mid_term_path,
            client=self.client,
            max_capacity=mid_term_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            use_embedding_api=self.use_embedding_api
        )
        self.user_long_term_memory = LongTermMemory(
            file_path=user_long_term_path,
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            use_embedding_api=self.use_embedding_api
        )

        # 初始化助手记忆模块
        self.assistant_long_term_memory = LongTermMemory(
            file_path=assistant_long_term_path,
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            use_embedding_api=self.use_embedding_api
        )

        # 初始化更新器和检索器
        self.updater = Updater(
            short_term_memory=self.short_term_memory,
            mid_term_memory=self.mid_term_memory,
            long_term_memory=self.user_long_term_memory,
            topic_similarity_threshold=self.mid_term_similarity_threshold,
            client=self.client, llm_model=self.llm_model)
        self.retriever = Retriever(
            mid_term_memory=self.mid_term_memory,
            long_term_memory=self.user_long_term_memory,
            assistant_long_term_memory=self.assistant_long_term_memory,  # Pass assistant LTM
            queue_capacity=retrieval_queue_capacity
        )


    def add_memory(self, user_input: str, agent_response: str, timestamp: str = None, metadata: dict = None):
        """
        添加新的交互记忆到系统中。会同时更新短期记忆和中期记忆，并根据需要更新长期画像。
        Push to STM, Insert to MTM
        
        Args:
            user_input: 用户输入
            agent_response: 助手响应
            timestamp: 时间戳
            metadata: 元数据
        """
        if not timestamp:
            timestamp = get_timestamp()
        
        qa_pair = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        self.short_term_memory.add_qa_pair(qa_pair)
        logger.info(f"MemoryOS: Added QA to STM. User: {user_input[:30]}...")

        if self.short_term_memory.is_full():
            logger.info("MemoryOS: STM full. Updating MTM...")
            self.updater.process_short_term_to_mid_term()

        self._trigger_profile_and_knowledge_update_if_needed()


    def _trigger_profile_and_knowledge_update_if_needed(self):
        """
        根据中期记忆的热度和内容，判断是否需要更新用户的长期画像和知识库。
        Heat > τ, Update to LPM
        """
        if not self.mid_term_heat_threshold:
            return
        if not self.mid_term_memory.heap:
            return

        # MTM 采用最小堆存储 heat
        neg_heat, sid = self.mid_term_memory.heap[0]
        cur_heat = -neg_heat
        
        if cur_heat >= self.mid_term_heat_threshold:
            session = self.mid_term_memory.sessions.get(sid)
            if not session:
                self.mid_term_memory.rebuild_heap()
                return

            # Page: {"user_input": ..., "agent_response": ..., "timestamp": ..., "analyzed": False, ...}
            unanalyzed_pages = [p for p in session.get("details", []) if not p.get("analyzed", False)]
            
            if unanalyzed_pages:
                logger.info(f"MemoryOS: Mid-term session {sid} heat ({cur_heat:.2f}) exceeded threshold. Analyzing {len(unanalyzed_pages)} pages for profile/knowledge update.")

                # 并行执行两个LLM任务：用户画像分析（已包含更新）、知识提取
                def task_user_profile_analysis():
                    existing_profile = self.user_long_term_memory.get_raw_user_profile(self.user_id)
                    if not existing_profile or existing_profile.lower() == "none":
                        existing_profile = "No existing profile."
                    return gpt_user_profile_analysis(unanalyzed_pages, self.client,
                                                     model=self.llm_model, existing_user_profile=existing_profile)

                def task_knowledge_extraction():
                    return gpt_knowledge_extraction(unanalyzed_pages, self.client, model=self.llm_model)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_profile = executor.submit(task_user_profile_analysis)
                    future_knowledge = executor.submit(task_knowledge_extraction)
                    try:
                        updated_user_profile = future_profile.result()
                        knowledge_result = future_knowledge.result()
                    except Exception as e:
                        logger.error(f"MemoryOS: Error during parallel LLM tasks: {e}")
                        return

                new_user_private_knowledge = knowledge_result.get("private")
                new_assistant_knowledge = knowledge_result.get("assistant_knowledge")

                if updated_user_profile and updated_user_profile.lower() != "none":
                    self.user_long_term_memory.update_user_profile(self.user_id, updated_user_profile, merge=False)
                    logger.info(f"MemoryOS: Updated user profile based on mid-term session {sid} analysis.")

                # 添加用户私人知识到用户 LTM
                if new_user_private_knowledge and new_user_private_knowledge.lower() != "none":
                    for line in new_user_private_knowledge.split("\n"):
                        if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                            self.user_long_term_memory.add_user_knowledge(line.strip())

                # 添加助手知识到助手 LTM
                if new_assistant_knowledge and new_assistant_knowledge.lower() != "none":
                    for line in new_assistant_knowledge.split("\n"):
                        if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                            self.assistant_long_term_memory.add_assistant_knowledge(line.strip())

                # 更新会话状态
                for p in session["details"]:
                    p["analyzed"] = True
                session["N_visit"] = 0
                session["L_interaction"] = 0
                session["H_segment"] = compute_segment_heat(session)
                session["last_visit_time"] = get_timestamp()
                self.mid_term_memory.rebuild_heap()     # 更新堆中热度值
                logger.info(f"MemoryOS: Completed profile and knowledge update for mid-term session {sid}. Heat reset.")

            else:
                logger.info(f"MemoryOS: Mid-term session {sid} heat ({cur_heat:.2f}) exceeded threshold but no unanalyzed pages found. Skipping profile update.")


    def get_response(self, query: str,
                     relationship_with_user: str = "friend", style_hint: str = "",
                     user_conversation_meta_data: dict = None) -> str:
        """
        检索相关的记忆内容，并生成一个综合的响应。
        
        Args:
            query: 用户查询
            relationship_with_user: 与用户的关系
            style_hint: 风格提示
            user_conversation_meta_data: 用户对话元数据
        
        Returns:
            生成的响应内容
        """
        logger.info("MemoryOS: Retrieving relevant memories for response generation...")
        
        # 检索相关记忆
        retrieval_results = self.retriever.retrieve_context(
            user_query=query, user_id=self.user_id
        )
        retrieved_pages = retrieval_results["retrieved_pages"]  # MTM Top-K pages
        retrieved_user_knowledge = retrieval_results["retrieved_user_knowledge"]  # LPM user knowledge
        retrieved_assistant_knowledge = retrieval_results["retrieved_assistant_knowledge"]  # LPM assistant knowledge

        # 构建短期记忆历史
        short_term_history = self.short_term_memory.get_all()
        history_text = "\n".join([
            f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')} (Time: {qa.get('timestamp', '')})"
            for qa in short_term_history
        ])

        # 构建中期记忆检索文本
        retrieval_text = "\n".join([
            f"【Historical Memory】\nUser: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}\nTime: {page.get('timestamp', '')}\nConversation chain overview: {page.get('meta_info','N/A')}"
            for page in retrieved_pages
        ])

        # 获取用户画像
        user_profile_text = self.user_long_term_memory.get_raw_user_profile(self.user_id)
        if not user_profile_text or user_profile_text.lower() == "none":
            user_profile_text = "No user profile available."

        # 构建用户知识背景
        user_knowledge_background = ""
        if retrieved_user_knowledge:
            user_knowledge_background = "\n【Relevant User Knowledge Entries】\n"
            for kn_entry in retrieved_user_knowledge:
                user_knowledge_background += f"- {kn_entry['knowledge']} (Recorded: {kn_entry['timestamp']})\n"
        background_context = f"【User Profile】\n{user_profile_text}\n{user_knowledge_background}"

        # 构建助手知识文本
        assistant_knowledge_text_for_prompt = "【Assistant Knowledge Base】\n"
        if retrieved_assistant_knowledge:
            for ak_entry in retrieved_assistant_knowledge:
                assistant_knowledge_text_for_prompt += f"- {ak_entry['knowledge']} (Recorded: {ak_entry['timestamp']})\n"
        else:
            assistant_knowledge_text_for_prompt += "No relevant assistant knowledge found for this query.\n"

        # 构建元数据文本
        meta_data_text_for_prompt = "【Current Conversation Metadata】\n"
        if user_conversation_meta_data:
            try:
                meta_data_text_for_prompt += json.dumps(user_conversation_meta_data, indent=2, ensure_ascii=False)
            except TypeError:
                meta_data_text_for_prompt += str(user_conversation_meta_data)
        else:
            meta_data_text_for_prompt += "None provided for this turn.\n"

        # 构建最终提示
        system_prompt_text = prompts.GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT.format(
            relationship=relationship_with_user,
            assistant_knowledge_text=assistant_knowledge_text_for_prompt,
            meta_data_text=meta_data_text_for_prompt
        )

        user_prompt_text = prompts.GENERATE_SYSTEM_RESPONSE_USER_PROMPT.format(
            history_text=history_text,          # STM
            retrieval_text=retrieval_text,      # MTM
            background=background_context,      # LPM
            relationship=relationship_with_user,
            query=query
        )

        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text}
        ]

        logger.info(f"MemoryOS: Constructed prompt for LLM:\nSystem Prompt:\n{system_prompt_text}\n\nUser Prompt:\n{user_prompt_text}\n")
        logger.info("MemoryOS: Calling LLM for response generation...")
        response_content = self.client.chat_completion(
            model=self.llm_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        # 添加新的记忆
        self.add_memory(user_input=query, agent_response=response_content, timestamp=get_timestamp())
        return response_content


    def __repr__(self):
        return f"<MemoryOS user_id='{self.user_id}' assistant_id='{self.assistant_id}' data_path='{self.data_storage_path}'>"