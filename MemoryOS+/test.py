import os
from memoryos import MemoryOS

# --- 基本配置 ---
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
API_KEY = ""  # 替换为您的API密钥
BASE_URL = "https://api.siliconflow.cn/v1"  # 使用 SiliconFlow 或其他兼容 OpenAI 的 API
DATA_STORAGE_PATH = "./simple_demo_data"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"  # API embedding 模型


def simple_demo():
    print("MemoryOS 简单演示")

    # 1. 初始化 MemoryOS
    print("正在初始化 MemoryOS...")
    try:
        memo = MemoryOS(
            user_id=USER_ID,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=DATA_STORAGE_PATH,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,

            short_term_capacity=7,                 # 短期记忆容量（超过则转入中期）
            mid_term_heat_threshold=5,             # 中期记忆热度阈值（超过则更新长期画像）
            retrieval_queue_capacity=10,           # 检索时返回的最大页面数
            long_term_knowledge_capacity=100,      # 长期知识库容量
            mid_term_similarity_threshold=0.5,     # 中期记忆相似度阈值（判断是否合并会话）

            embedding_model_name=EMBEDDING_MODEL,  # 使用 API embedding 模型
            use_embedding_api=True  # 启用 API embedding 调用
        )
        print("MemoryOS 初始化成功！\n")

    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 2. 添加具有“局部话题连贯性”的记忆
    print("正在添加记忆...")

    # ===== 话题簇 1：日本北海道旅行规划 =====
    print("\n--- 话题簇 1 写入中 ---")
    memo.add_memory(
        user_input="我计划安排一个6天的旅行，打算去日本北海道。",
        agent_response="北海道冬天非常美，尤其是看雪！你打算具体去哪几个城市呢？"
    )
    memo.add_memory(
        user_input="目前规划了札幌、小樽和登别这几个地方。",
        agent_response="经典的道央路线！小樽的运河很浪漫，登别的地狱谷温泉也非常出名。"
    )
    memo.add_memory(
        user_input="对，后面如果时间充裕，我还想去洞爷湖和函馆看看。",
        agent_response="函馆的百万美元夜景绝对不容错过，洞爷湖的风景也很宜人，祝你旅途愉快！"
    )

    # ===== 话题簇 2：人工智能科研探讨 =====
    print("\n--- 话题簇 2 写入中 ---")
    memo.add_memory(
        user_input="最近我一直在阅读人工智能隐私和安全相关的文献。",
        agent_response="这是个非常核心的研究领域，你主要关注联邦学习还是差分隐私？"
    )
    memo.add_memory(
        user_input="我主要研究联邦学习，同时也对大模型强化学习（RL4LLM）很感兴趣。",
        agent_response="RL4LLM是当前大模型对齐的关键技术，结合联邦学习能有效解决数据孤岛问题。"
    )
    memo.add_memory(
        user_input="是的，我还准备把Agentic-RL应用到医疗大模型的后训练中去。",
        agent_response="非常有价值的研究方向！Agentic-RL可以让医疗大模型在诊断和推理时具备更强的自主规划能力。"
    )

    # ===== 话题簇 3：音乐与算法日常 =====
    print("\n--- 话题簇 3 写入中 ---")
    memo.add_memory(
        user_input="科研累了，晚上听了听方大同的歌放松一下。",
        agent_response="方大同的R&B和Soul风格确实非常有律动感，很适合放松心情。"
    )
    memo.add_memory(
        user_input="听完歌我打算刷两道算法题，复习一下快速排序和堆排序。",
        agent_response="劳逸结合得很好！这两种排序算法是面试常考的基础算法，祝你复习顺利。"
    )

    # 3. 测试记忆检索
    # 我们故意提问一个需要跨越上下文的话题，看看系统能否精准检索出特定话题簇的内容
    test_query = "你能结合我之前的讨论，帮我总结一下我接下来的旅行计划吗？"
    print(f"\n用户: {test_query}")

    response = memo.get_response(query=test_query)

    print(f"助手: {response}")


if __name__ == "__main__":
    simple_demo()