# 用 Python 学会一个最小 RAG

这个项目只有四个 Python 业务文件，完成“外部资料入库 → 理解问题 → 检索 → 根据资料回答”。
使用命令行交互，支持 UTF-8 的 `.txt` 和 `.md` 文件，附有三份虚构的中文产品资料。

## 1. 先理解两条流程

```text
入库（资料变化时运行）
docs/ 下的文件 → 按字符切块 → 向量模型 → Chroma 本地向量库

问答（每次提问时运行）
用户问题 → 聊天模型改写检索问题 → 向量模型 → 搜索相近的资料片段
                                                    ↓
                            原始问题 + 资料片段 → 聊天模型 → 带来源的回答
```

- **聊天模型（LLM）**：理解用户想查什么、阅读资料并生成回答。
- **向量模型（Embedding）**：把文字转换成一组数字，用来比较语义是否接近。
- **向量数据库（Chroma）**：保存向量、对应原文和来源，按向量距离查找近邻。

入库不训练或修改大模型。回答时，程序只是把查到的几段原文附在问题旁边。
这里的“意图解析”是一次简单的问题改写，例如把“我不小心删了笔记，咋找回来？”改写成“星河笔记误删笔记的恢复步骤”。
为了保留用户原意，生成最终回答时仍然传入原始问题。每次 `ask` 是独立问答，没有聊天历史。

## 2. 安装并配置

使用 Python 3.10 或以上版本。在项目目录运行，以下命令适用于 macOS / Linux：

```bash
cd /Users/allen/code/myRAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

如果本地已经有 `.venv`，直接激活即可；`cp -n` 会保留已有的 `.env`。

打开 `.env`，填写两个 Key。默认示例使用硅基流动，可以先用同一个 Key 填两次：

```dotenv
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=填入你的Key
LLM_MODEL=deepseek-ai/DeepSeek-V3.2

EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=填入你的Key
EMBEDDING_MODEL=BAAI/bge-m3
```

模型名称是示例，以你的服务商账户实际可调用的模型为准。接口形式见[硅基流动聊天文档](https://docs.siliconflow.cn/docs/userguide/capabilities/text-generation)和[向量文档](https://docs.siliconflow.cn/docs/api/embeddings-post)。
模型调用会向所配置的服务发送问题或资料文本，并按该服务的规则计费。本地向量库保存在 `storage/`。

配置从项目目录的 `.env` 读取；同名的终端环境变量优先。`.env` 已列入 `.gitignore`，不要把真实 Key 写进代码。

## 3. 先入库，再提问
```bash
# 读取 docs/，生成向量，并保存到本地
python main.py ingest

# 只做向量搜索：先观察是否能找对原文，不调用聊天模型
python main.py search "误删笔记怎么恢复？"

# 完整 RAG：解析问题 → 搜索资料 → 根据资料回答
python main.py ask "我不小心删了笔记，该怎么找回来？"

# 资料不足的例子：示例资料没有价格，应明确表示信息不足
python main.py ask "星河笔记团队版每年多少钱？"
```

### 3.1 使用网页界面

网页提供聊天问答、行内引用预览、资料上传、知识库同步和原文件预览：

```bash
streamlit run app.py
```

回答中的 `[1]`、`[2]` 是可点击引用。点击后会在引用位置打开可缩放的预览框，
自动定位并高亮原文件中的对应行；Markdown 文件也可以切换到渲染预览。
上传、修改或删除原文件后，需要点击“同步知识库”，问答才会使用最新内容。

当前聊天记录只在浏览器会话中展示，每次 `RAG.ask()` 仍是独立问答，模型不会读取上一轮对话。

`ask` 会显示改写后的检索问题、命中的原文、文件名、块编号、距离，以及最终答案。
例如，恢复笔记的回答应包含“设置 → 备份与恢复 → 选择删除前的备份 → 恢复笔记”，并用 `[1]` 等编号指向实际检索片段。具体措辞和来源顺序由模型及检索结果决定。

`ingest` 使用文件级增量同步。添加、修改或删除 `docs/` 中的文件后，再运行一次 `ingest` 才会生效；未变化的文件不会重复生成向量。空目录或只有空文件时会报错并保留原索引。

### 3.2 资料的增量入库逻辑

程序扫描 `docs/` 下的 `.md`、`.txt`，计算 SHA-256 内容哈希，再根据“相对路径 + 内容哈希”与数据库比较：

| 状态 | 判断 | 处理 |
|---|---|---|
| 新增 | 路径未入库 | 切块、生成向量并写入 |
| 修改 | 路径相同，内容哈希变化 | 删除该文件全部旧片段，重新切块并写入 |
| 删除 | 数据库中存在，但目录中不存在 | 删除该文件全部记录 |
| 未变化 | 路径和哈希均相同 | 跳过 |

每个片段保存原文、Embedding 向量、文件相对路径、文件内容哈希和片段编号。

为保证安全，程序先完成新增和修改文件的向量生成，再更新数据库；如果 Embedding 调用失败，保留原数据。

Embedding 模型、服务地址、`CHUNK_SIZE` 或 `CHUNK_OVERLAP` 变化时，程序自动全量重建；聊天模型、`TOP_K` 或提示词变化不需要重新入库。

暂不识别文件改名：改名会被视为“删除旧文件并新增新文件”，不会留下重复数据，但会重新生成向量。

### 3.3 注意事项

```bash
# 读取 docs/，生成向量，并保存到本地
python main.py ingest
```

入库之后，数据库默认保存在 `storage/`。修改、增加或删除资料后，需要重新运行 `ingest`；程序会自动判断需要增量更新的文件。

## 4. 按这个顺序读代码

| 文件 | 作用 | 先关注什么 |
| --- | --- | --- |
| `main.py` | 接收命令、显示结果 | `ingest`、`search`、`ask` 分别做什么 |
| `rag.py` | 完整 RAG 逻辑 | 先读 `ask()` 的三个步骤，再读 `ingest()` |
| `models.py` | 封装两个模型接口 | `chat()` 返回文字，`embed()` 返回向量 |
| `config.py` | 读取 `.env` | 聊天和向量配置相互独立 |

可以把 `ask()` 的主体理解成：

```python
query = models.chat(改写指令, 用户原问题)
sources = search(query)  # 内部调用 embed()，然后查询 Chroma
answer = models.chat(回答指令, 用户原问题和检索原文)
```

真实代码还会检查空问题、未入库、向量配置不一致等常见错误。

## 5. 更换模型、尝试自己的资料

**换聊天模型**：修改 `.env` 中的 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`，再运行 `ask`，无需重新入库。
新服务需要支持 OpenAI 兼容的 `/chat/completions` 接口。这里使用 `openai` Python 包作为客户端，实际请求发往你配置的地址，不要求使用 OpenAI 的模型。

**换向量模型**：修改三个 `EMBEDDING_*` 配置，然后重新运行 `ingest`。服务需要支持 `/embeddings`。
聊天和向量可以来自不同服务商；提供聊天接口不代表提供向量接口。更换向量模型或地址后必须重新入库，即使新旧向量维度相同，也不能直接比较。

不兼容这两个接口的服务，集中修改 `models.py` 的 `chat()` / `embed()`，保持返回值形式即可。
服务商专属参数也在这里增加。示例聊天请求只发送通用的 `model` 和 `messages`；如果需要限制输出长度，可按所选模型的文档添加它支持的参数，例如 `max_tokens` 或 `max_completion_tokens`，不要假定所有模型接受相同参数。

建议每次只改一件事：

1. 放一份自己的 `.md` 进 `docs/`，入库后询问其中的事实。
2. 在 `.env` 中调整 `CHUNK_SIZE`（每块字符数）和 `CHUNK_OVERLAP`（重叠字符数），重新入库，观察检索片段。
3. 调整 `TOP_K`（取回片段数，默认 3），直接重新提问，观察回答是否获得了更完整的依据。
4. 换一个聊天模型，对同样资料和问题比较回答。
5. 最后尝试换向量模型，并重新入库，比较搜索结果。

初版按字符切块，字符数不是 token 数，长句可能在块边界被截断。向量搜索会返回最接近的片段，即使它们并不能回答问题；距离越小表示越相似，**距离不是准确率**。
回答提示词要求资料不足时直说，但不保证模型永远遵守，因此展示原文是核对答案的一部分。这个版本不设置通用相似度阈值。

以后需要 PDF，可以在 `ingest()` 的文件读取位置增加文本提取；需要更好的检索，可以在 `search()` 增加关键词搜索或重排。当前实现适合小规模学习资料，切块和向量会暂存在内存，不做并发入库。

## 常见问题

- **没有填配置**：按错误提示填写 `.env`；入库和 `search` 只需要向量配置，`ask` 两组都需要。
- **HTTP 401 / 403**：检查 Key、模型权限与账户状态。
- **HTTP 404**：检查服务地址、模型名称、对应端点是否存在。
- **HTTP 400**：检查模型支持的参数与输入长度，必要时减小切块大小。
- **HTTP 429**：检查调用额度和频率限制，稍后再试。
- **资料已经改了但答案没变**：重新执行 `ingest`。
- **文件无法读取**：确认它是 UTF-8 编码的 `.txt` 或 `.md`。

## 验证与官方参考

```bash
python -m unittest discover -s tests -v
```

测试使用临时目录中的真实 Chroma 和模拟模型响应，不需要 Key，也不会调用真实模型。它们验证程序流程；真实模型的连接和回答质量需要填写自己的 Key 后运行上述命令验证。

接口参考：[OpenAI Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)、[OpenAI Embeddings](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)、[Chroma 集合](https://docs.trychroma.com/docs/collections/manage-collections)、[Chroma 查询](https://docs.trychroma.com/docs/querying-collections/query-and-get)。
