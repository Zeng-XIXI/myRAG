"""判断用户输入应该如何处理。"""
# 从dataclasses模块导入dataclass装饰器，用于快速定义不可变数据类
from dataclasses import dataclass
# 从typing导入Literal，用于限定变量只能是指定的字面量值
from typing import Literal

# 定义路由名称的字面量类型，仅允许chat/unclear/rag三个取值
RouteName = Literal["chat", "unclear", "rag"]

# 定义冻结（不可修改）的数据类，保存单次用户输入的路由判定结果
@dataclass(frozen=True)
class RouteDecision:
    """一次用户输入的路由判断结果。"""
    # 路由类型，只能是上面RouteName规定的三个值
    route: RouteName
    # 是否需要重写用户问题，用于后续检索优化
    rewrite: bool
    # 本次路由判定的原因说明，用于日志/调试
    reason: str
    # 直接回复用户的答案，只有闲聊/无法识别场景才会填充，rag场景为None
    answer: str | None = None

# 闲聊话术映射字典：key是标准化后的闲聊语句，value是预设回复
CASUAL_REPLIES = {
    "你好": "你好！你可以向我询问知识库中的内容。",
    "您好": "您好！你可以向我询问知识库中的内容。",
    "嗨": "嗨！有什么知识库问题想了解吗？",
    "hi": "Hi！你可以向我询问知识库中的内容。",
    "hello": "Hello！你可以向我询问知识库中的内容。",
    "谢谢": "不客气！如果还有问题，可以继续提问。",
    "感谢": "不客气！如果还有问题，可以继续提问。",
    "再见": "再见！",
}

# 疑问关键词元组，包含代表提问的中文词汇，用来识别用户是否在提问题
QUESTION_WORDS = (
    "什么",
    "如何",
    "怎么",
    "为什么",
    "为何",
    "哪里",
    "是否",
    "能否",
    "多少",
    "哪些",
)

# 指代模糊词汇元组，这类词依赖上文上下文，单独一句话无法判断指代对象
AMBIGUOUS_WORDS = (
    "这个",
    "那个",
    "它",
    "上面",
    "下面",
    "之前",
    "刚才",
)

def normalize_question(question: str) -> str:
    """清理两端空白和常见句末标点。"""
    # 去除字符串首尾空白、转小写、移除句末中英文标点，做文本标准化
    return question.strip().lower().rstrip("。！？!?，,")

def classify_question(question: str) -> RouteDecision:
    """判断输入是闲聊、信息不足，还是知识库问题。"""
    # 对用户原始输入做标准化处理
    normalized = normalize_question(question)

    # 如果标准化后的文本命中闲聊字典，路由走chat，直接返回预设回答
    if normalized in CASUAL_REPLIES:
        return RouteDecision(
            route="chat",
            rewrite=False,
            reason="识别为日常对话，无需查询知识库",
            answer=CASUAL_REPLIES[normalized],
        )
    # 处理空输入：用户只输入空格/标点，无有效文字
    if not normalized:
        return RouteDecision(
            route="unclear",
            rewrite=False,
            reason="用户没有输入有效内容",
            answer="请输入你想查询的具体问题。",
        )
    # 检测是否包含模糊指代词，命中则判定为问题不明确，unclear路由
    if any(word in normalized for word in AMBIGUOUS_WORDS):
        return RouteDecision(
            route="unclear",
            rewrite=False,
            reason="问题包含不明确的指代，但当前对话不支持多轮上下文",
            answer="请补充你所指的具体对象，例如文件名称、功能名称或配置项。",
        )
    # 判断文本长度>=40字符，标记为需要改写问题，用于rag检索优化
    should_rewrite = len(normalized) >= 40
    # 进入RAG知识库检索路由，返回判定结果
    return RouteDecision(
        route="rag",
        rewrite=should_rewrite,
        reason=(
            "问题较长，先改写为简洁的检索问题"
            if should_rewrite
            else "问题表达较明确，直接使用原问题检索"
        ),
    )
