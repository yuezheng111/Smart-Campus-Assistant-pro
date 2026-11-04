# 校园智助 · 规模化技术方案

> 场景：1 万学生、峰值 1000 人同时在线、1 万份文件、管理端为主体、上传或修改后学生端准实时可见。
> 这份文档只回答一个问题：**现有单机实现要变成能扛住这个量级的系统，哪些地方必须改，改的代价和边界在哪。**

---

## 一、先做规模换算（所有结论的地基）

不换算就没法谈取舍。下面全部是**假设值**，上线前必须用压测校准，但它们决定了架构方向。

### 1.1 在线数 ≠ QPS

「1000 人同时访问」不等于 1000 QPS。学生端的行为是：登录 → 翻列表 → 偶尔提问，提问是低频动作。

| 场景 | 假设 | 推导 QPS |
|---|---|---|
| 日常 | 1000 在线，5 分钟内人均问 1 次 | 1000 / 300 ≈ **3.3** |
| 报名截止前 | 1000 在线，3 分钟问 1 次 | 1000 / 180 ≈ **5.5** |
| 热门通知发布瞬间 | 10% 的人同一分钟发问 | ≈ **1.7/s 起，短时冲 **10–20** |

**设计目标取：稳态 20 QPS、峰值 100 QPS（持续 1 分钟）。** 再高就不是校园场景了。

### 1.2 在途并发 = QPS × 响应时间

单问端到端约 3 秒（含 LLM）→ `100 QPS × 3s = 300 个请求同时在途`。

这 300 个在途请求各自占什么资源，决定了瓶颈在哪：

| 资源 | 单请求占用 | 300 在途总计 | 结论 |
|---|---|---|---|
| Python 协程 | ~几 KB | 几 MB | 可忽略 |
| OS 线程（若用同步） | 8MB 栈 | **2.4 GB** | **不可接受** |
| MySQL 连接 | 1 | 300（若持锁式占用） | **需要连接池复用** |
| 外部 LLM 调用 | 1 个 HTTP 长连接 | 300 | **受供应商配额限制** |

**第一个结论：同步模型撑不住。** 300 个线程不现实，必须全链路 async。

### 1.3 向量库内存推导

| 项 | 计算 | 结果 |
|---|---|---|
| chunk 总数 | 1 万份 × 平均 10 个 | **10 万** |
| 纯向量 | 10 万 × 1024 维 × 4 B | **~400 MB** |
| HNSW 图开销 | 约 +30% | +120 MB |
| 正文 payload | 10 万 × 600 字 × 3 B | ~180 MB |
| **合计** | | **约 700 MB – 1 GB** |

**第二个结论：单机装得下，但装不下第二份。** 如果 uvicorn 开 4 个 worker，每个 worker 各自加载一份嵌入式向量库 → 3–4 GB，且互不可见。这就是必须把向量库**服务化**的根本原因，不是"想上分布式"。

> 现有 `app/rag/vectorstore.py:141` 的注释其实已经写下了这个判断：「真到了那个量级，该换向量库了（Chroma 是嵌入式单机方案）」。这份方案就是把那句话落地。

---

## 二、架构总览

```
学生端（1000 在线）        管理端（对话改文件）
        └────────┬────────────────┘
                 ▼
        ┌─────────────────────────────┐        ┌──────────────┐
        │ 接入：Nginx / 限流 / 鉴权      │        │ 解析 Worker 池 │
        ├─────────────────────────────┤◄──任务──┤ OCR / 切分 /  │
        │ 应用：FastAPI async N worker  │        │ embedding     │
        │  Agent 编排 · 检索 · 缓存      │        └──────┬───────┘
        ├─────────────────────────────┤◄──轮询──┐       │
        │                               │        │       │
        └───────┬────────┬────────┬─────┘   ┌────┴───┐   │
                ▼        ▼        ▼         │ Outbox │◄──┘
             MySQL    Qdrant    Redis       └────────┘
```

四层职责：

| 层 | 职责 | 关键约束 |
|---|---|---|
| 接入 | TLS、限流、鉴权、静态资源 | 无状态，可水平扩 |
| 应用 | Agent 编排、检索、生成、缓存 | **全 async**，无本地状态 |
| 异步 | 解析、embedding、索引同步 | 独立进程池，不阻塞 web |
| 存储 | MySQL（真相源）/ Qdrant（向量）/ Redis（缓存配额）/ 文件盘 | 各自独立扩缩 |

**一条设计原则贯穿全篇：MySQL 是唯一真相源，Qdrant 是可重建的派生索引，Redis 是可丢弃的缓存。** 这个层级关系决定了故障时谁先降级、谁能丢数据。

---

## 三、数据存储与索引

### 3.1 量级判断：不需要分库分表

| 表 | 行数 | 判断 |
|---|---|---|
| `student` | 1 万 | 极小 |
| `document` | 1 万 | 极小 |
| `competition` / `exam` | 千级 | 极小 |
| `ingest_task` | 1 万 + 历史 | 需归档 |
| `agent_session` | 1 万活跃 | 需 TTL |
| `chunk`（若落库） | 10 万 | 单表无压力 |

**明确结论：这个量级单库单表绰绰有余，分库分表是过度设计。** 唯一需要处理的是「历史数据增长」——`agent_session`、`ingest_task`、请求日志按月归档即可。

写下来是为了防止重构时被"万级规模"这几个字带偏。

### 3.2 表结构（MySQL 5.7 兼容）

新增三张表，**不动现有 `document` / `competition` / `exam`**（`init_db.py` 会 DROP，加列等于每次初始化都要重建）。

```sql
-- 学生账号：登录 + 画像（画像字段已存在于现有 student 表，这里只补认证）
ALTER TABLE student ADD COLUMN password_hash VARCHAR(128) DEFAULT NULL;
ALTER TABLE student ADD COLUMN last_login_at DATETIME DEFAULT NULL;
CREATE UNIQUE INDEX uk_student_no ON student(student_no);
CREATE INDEX idx_student_org ON student(college, major);

-- 入库任务：上传即返回，后台解析
CREATE TABLE ingest_task (
  id           BIGINT AUTO_INCREMENT PRIMARY KEY,
  file_path    VARCHAR(512) NOT NULL,
  content_hash CHAR(64)     NOT NULL,
  op           ENUM('upsert','delete') NOT NULL DEFAULT 'upsert',
  status       ENUM('pending','claimed','parsing','indexing','done','failed')
               NOT NULL DEFAULT 'pending',
  worker_id    VARCHAR(64)  DEFAULT NULL,
  attempts     TINYINT      NOT NULL DEFAULT 0,
  error        TEXT         DEFAULT NULL,
  created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  claimed_at   DATETIME     DEFAULT NULL,
  finished_at  DATETIME     DEFAULT NULL,
  KEY idx_pick (status, id),
  KEY idx_path (file_path)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 索引同步 outbox：解决 MySQL 与 Qdrant 双写不一致
CREATE TABLE index_outbox (
  id         BIGINT AUTO_INCREMENT PRIMARY KEY,
  doc_id     BIGINT      NOT NULL,
  file_path  VARCHAR(512) NOT NULL,
  op         ENUM('upsert','delete','meta_only') NOT NULL,
  payload    JSON        DEFAULT NULL,
  status     ENUM('pending','done','failed') NOT NULL DEFAULT 'pending',
  attempts   TINYINT     NOT NULL DEFAULT 0,
  created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_pending (status, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 会话：替代前端 chatHistory
CREATE TABLE agent_session (
  session_id  CHAR(36) PRIMARY KEY,
  student_no  VARCHAR(32) NOT NULL,
  turns       JSON        DEFAULT NULL,
  entities    JSON        DEFAULT NULL,
  updated_at  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY idx_student (student_no, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 知识库版本（单行）：驱动缓存失效
CREATE TABLE kb_version (
  id         TINYINT PRIMARY KEY DEFAULT 1,
  version    BIGINT  NOT NULL DEFAULT 1,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 3.3 索引设计要点

```sql
-- 默认检索路径：现行 + 原件在 + 按时间倒序
CREATE INDEX idx_doc_live ON document(status, file_missing_at, publish_date);
-- 业务过滤：类型/年份/学院
CREATE INDEX idx_doc_filter ON document(doc_type, year, college);
-- 生命周期：同一族的最新版本
CREATE INDEX idx_doc_family ON document(doc_family, version);
-- 幂等基础：一份文件只有一行
CREATE UNIQUE INDEX uk_doc_path ON document(file_path);

-- 竞赛：按届次+类别查、按报名截止查
CREATE INDEX idx_comp_year ON competition(year, category);
CREATE INDEX idx_comp_window ON competition(signup_end);
```

**为什么 `(status, file_missing_at, publish_date)` 是这个顺序**：等值列在前、排序列在后，才能既走过滤又避免 filesort。反过来写会让每次列表查询都额外排序。

**Qdrant 侧**：`doc_type / college / year / status / file_missing / is_current` 这六个字段建 payload index。没有 payload index，过滤会退化成全量扫 payload。

---

## 四、并发与负载处理

### 4.1 四个具体瓶颈（都能定位到代码行）

| # | 瓶颈 | 证据 | 影响 |
|---|---|---|---|
| 1 | **同步函数阻塞线程池** | `app/api/main.py:204` 是 `def chat()`，FastAPI 丢到 anyio 线程池（默认 40） | 第 41 个并发请求开始排队 |
| 2 | **MySQL 连接池默认 5+10** | `app/db.py:35` 的 `create_engine` 没设 `pool_size` | 15 个并发后排队，长事务会拖死全局 |
| 3 | **Chroma 嵌入式单进程** | `app/rag/vectorstore.py:32` 全局单例 + 文件锁 | 多 worker 下内存翻倍且写入互不可见 |
| 4 | **LLM 供应商配额** | 无并发控制，全量透传 | 峰值直接被限流，大批 5xx |

第 4 条才是真正的天花板，前三者都是"自己把自己卡住"。

### 4.2 解法

**① 全链路 async**

```python
@app.post("/api/chat")
async def chat(req: ChatRequest):
    ...
    result = await run_agent_async(...)   # 内部 await llm.ainvoke / await tool.ainvoke
```

LangChain 的工具和模型都有 `ainvoke`，改造是机械的但必须做全 —— 混一个同步调用，整个协程就被那个线程卡住。300 在途请求用协程是几 MB，用线程是 2.4 GB。

**② 连接池按 worker 数算**

```python
create_engine(url, pool_size=10, max_overflow=5, pool_pre_ping=True, pool_recycle=1800)
# 4 worker × 15 = 60 < MySQL max_connections(默认 151)
```

`pool_pre_ping` 保留（现有代码已经加了，是对的），`pool_recycle` 从 3600 降到 1800 —— 云数据库常见的 8 小时断连不是唯一杀手，NAT 超时更早。

**③ 向量库服务化**

换成 Qdrant（独立进程，HTTP/gRPC）。应用层变成无状态，worker 数量随便加，写入立即可见（这正是"实时更新"的前提）。

**④ LLM 并发闸门 + 分层降级**

```python
LLM_SEMAPHORE = asyncio.Semaphore(40)     # 按供应商实际配额定，不是拍脑袋

async def guarded_generate(...):
    try:
        async with asyncio.timeout(8):     # 排队超时
            async with LLM_SEMAPHORE:
                return await llm.ainvoke(...)
    except (TimeoutError, RateLimitError):
        return degrade_to_retrieval_only(...)   # 降级：只给检索结果
```

**这是"100 QPS 可行性"的关键**：不可能 100 QPS 全部走 LLM。必须让"必须 LLM"的请求降到 40% 以下，靠缓存 + 降级 + 非 LLM 路径分流。

### 4.3 三层缓存

| 层 | 内容 | TTL | 失效 |
|---|---|---|---|
| L1 进程内 LRU | 热门问答、检索结果 | 60s | 进程重启 / kb_version 变 |
| L2 Redis | 跨 worker 共享问答 | 10min | **kb_version 变** |
| L3 预计算 | 「国奖条件」这类 Top50 高频问题的完整答案 | 至下次入库 | kb_version 变 |

**缓存 key 必须包含三样**：

```
key = sha1(normalize(question) + kb_version + student_context_hash)
```

- `kb_version`：文件一改，全库缓存自然失效，**不需要精确失效逻辑**（见 §6）。
- `student_context_hash`：同一个问题，大二和大四答案不同（planning 类）。少了这一项，就会出现"大二学生看到大四的规划建议"。
- `normalize`：去空格、标点、语气词，让"国奖几个名额？"和"国奖名额多少"命中同一条。

### 4.4 限流

| 对象 | 策略 | 理由 |
|---|---|---|
| 单学生 | 10 次/分钟 | 防脚本刷，正常学生够用 |
| 全局 LLM | 信号量 40 | 供应商配额上限 |
| 排队长度 | >200 直接 429 + 提示稍后 | 排队比超时更伤体验 |
| 上传 | 5 份/次，25MB/批（沿用现有） | 已有实现，不动 |

---

## 五、文件的异构解析与入库

### 5.1 现状问题

`POST /api/admin/upload` 里同步调 `ingest_paths()`，请求会挂几十秒（大 PDF + embedding）。1 万份批量导入不可能这么干。

### 5.2 流水线

```
上传 → 落盘 → 写 ingest_task(pending) → 立即返回 task_id
                                      ↓
              Worker 池（N=CPU 核数，独立进程）
               领取 → 按格式路由解析 → 切分 → embedding → 写 Qdrant → 更新 MySQL → 递增 kb_version → done
                                                                     ↓失败
                                                              attempts++ 重试 ≤3
```

**状态机**：`pending → claimed → parsing → indexing → done | failed`
每一步落库，所以「卡在哪一步」永远可查 —— 这是运维上最值钱的字段。

### 5.3 任务领取：MySQL 5.7 没有 SKIP LOCKED

现有环境是 MySQL **5.7.44**，`SELECT ... FOR UPDATE SKIP LOCKED` 是 8.0 的特性，用不了。可用的是「原子 UPDATE + 回查」：

```sql
-- 1) 原子领取：受影响的只有一行，靠行锁 + 状态条件保证互斥
UPDATE ingest_task
SET status='claimed', worker_id=:wid, claimed_at=NOW(), attempts=attempts+1
WHERE status='pending' ORDER BY id LIMIT 1;

-- 2) 回查自己领到了哪一行
SELECT * FROM ingest_task WHERE worker_id=:wid AND status='claimed';
```

两条语句必须在一个事务里。这个模式在 5.7 上是标准做法，比"先 SELECT 再 UPDATE"安全得多 —— 后者两个 worker 会抢到同一行。

**僵尸任务回收**：`claimed_at` 超过 10 分钟仍是 `claimed` 的，放回 `pending`。Worker 被 kill 时不至于让任务永远卡住。

### 5.4 异构格式路由

| 格式 | 解析器 | 耗时 | 说明 |
|---|---|---|---|
| `.md` / `.txt` | 直读 | ms | 现有 |
| `.docx` | python-docx | 0.1s | 现有 |
| `.xlsx` / `.csv` | pandas → markdown 表 | 0.1s | **新增，优先级最高** |
| 文本层 PDF | pdfplumber | 0.1–0.5s/页 | 现有，但表格会拍平 |
| 表格 PDF | pdfplumber `extract_tables` | 0.3s/页 | **新增，表格单独成 chunk** |
| 扫描件 PDF | OCR（PaddleOCR） | **1–3s/页** | 成本大头 |
| 图片 | OCR | 1–3s | 低优先级 |
| `.pptx` | python-pptx | 0.2s | 中优先级 |

**OCR 的成本要算清楚**：1 万份 × 平均 20 页 × 2s = **11 万秒 ≈ 31 核·小时**。单核跑要 31 小时，8 核并行也要 4 小时。所以：

> **不做"全量 OCR"，做"按需 OCR"。** 上传时检测有无文本层；无文本层的文件标记 `needs_ocr=true`，进低优先级队列，先保证"已知可解析的 90% 先可用"，再慢慢啃剩下的 10%。

这是典型的「可用性优先于完整性」取舍。

### 5.5 embedding 成本与限速

```
10 万 chunk ÷ 批大小 10 = 1 万次 API 调用
1 万次 × 0.2s ≈ 33 分钟（串行）；10 路并发 ≈ 3.5 分钟
```

必须限速：DashScope 有 QPS 限制，超了直接 429。Worker 内加令牌桶，批量导入时整体降速，避免把线上查询的配额挤掉。

**幂等**：靠 `content_hash` 跳过未变文件（现有 `_unchanged_files` 已实现），只处理真正变了的。1 万份全量重灌和"只改 3 份"的成本差三个数量级。

---

## 六、实时更新的同步机制

### 6.1 先定义"实时"

**秒级可见**（管理端保存后，学生下一次请求就能看到），不做推送。理由：校园文件不是股票行情，用户不会盯着看有没有更新；引入 WebSocket/SSE 广播的复杂度换不来体验。

### 6.2 三条链路，一条比一条重

| 场景 | 链路 | 延迟 |
|---|---|---|
| **改元数据**（标题/类型/学院/年份） | MySQL UPDATE → `index_outbox(meta_only)` → Worker 改 Qdrant payload → `kb_version++` | 秒级 |
| **替换正文**（新版本文件） | 新版入库 → 旧版 `status=superseded` → 删旧 chunk → 写新 chunk → `kb_version++` | 秒级 |
| **批量重建**（换 embedding 模型/改切分策略） | 新 collection 双写 → 全量灌完 → 原子切 alias | 分钟–小时，期间学生无感 |

### 6.3 双写不一致：用 outbox，不用两阶段提交

改一份文件要同时写 MySQL 和 Qdrant。顺序错了就出问题：

- 先写 Qdrant 后写 MySQL 失败 → 向量库里有孤儿 chunk，学生搜得到、点开 404。
- 先写 MySQL 后写 Qdrant 失败 → 学生搜不到刚上传的文件，管理员以为传失败了。

**outbox 模式**：MySQL 业务更新和 outbox 记录**在同一个事务**里提交，Worker 异步消费 outbox 去更新 Qdrant。

```python
with engine.begin() as conn:          # 同一个事务
    conn.execute(update_document_sql)
    conn.execute(insert_outbox_sql)   # 要么都成功，要么都回滚
# Worker 消费 outbox；失败重试，attempts 超限进人工队列
```

效果：**不会出现"库里有、搜不到"或"搜得到、点不开"**。代价是一次更新要到秒级才完全生效 —— 这正是我们接受的定义。

### 6.4 缓存失效：用版本号，不用精确失效

`kb_version` 单调递增。任何影响学生可见内容的变更都 `version++`，缓存 key 带着它，**整库缓存自然失效**。

为什么不做精确失效（只清受影响文件的缓存）：
- 精确失效要维护"某文件被哪些缓存 key 引用"的反向索引，复杂且容易漏。
- 漏掉的后果很严重：学生看到已废止的旧政策，还以为是最新的。
- 代价是每次更新掉一小部分缓存，但更新是低频事件（每天几十次），冷启动压力可接受。

**这是一次明确的取舍：用「实现简单 + 一定正确」换「少量缓存重算」。**

### 6.5 一致性读

检索时只认 `is_current=1` 的 chunk（对应 MySQL 的 `status=active`）。替换正文的过程中，新旧 chunk 同时在库，但只有 `is_current` 的那份参与召回 —— 学生不会看到"一半新一半旧"。

这比"先删后写"安全：中途失败时旧数据还在，服务不中断。

---

## 七、管理端：对话式文件调整

### 7.1 为什么管理端是主体

学生端的动作空间是收敛的（找文件、查信息、问规划），枚举得出来。管理端的动作空间是发散的：查、改、批量改、废止、对账、重灌，组合无穷。**发散的动作空间才需要 Agent，收敛的用 Workflow 就够。**

这也解释了现状的合理性：管理端是 ReAct 循环，学生端是固定分支 —— 不是偷懒，是匹配问题结构。

### 7.2 权限边界（硬规则）

| 操作 | 模型能做 | 必须人做 |
|---|---|---|
| 查文件、查统计、查对账 | ✅ 直接执行 | — |
| 改元数据 | ✅ 生成变更单 | ✅ 点确认才生效 |
| 改正文 | ❌ | ✅ 只能"上传新版本 + 废止旧版" |
| 删除文件 | ❌ | ❌ 根本不提供（只标记缺失） |
| 批量操作 | ✅ 生成批量变更单 + dry-run 预览 | ✅ 确认后分批执行 |

**「不改正文」是有意的**：正文是证据。改了正文就无法回答"这份文件原文到底写了什么"。要修内容就发新版本 —— 这既符合公文实践，也保住了审计链。

### 7.3 对话调整的三条链路

```
① 单条定向：「把 12 号那份的标题改成 X」
   list/get 确认 id → propose_metadata_change → 变更单 → 人确认 → apply → outbox → kb_version++

② 条件批量：「把 2023 年所有奖学金通知的学院改成全校」
   list(status=active, doc_type=奖学金, year=2023) → 命中 17 条 → dry-run 预览 17 行 diff
   → 生成批量变更单 → 人确认 → 后台分批 apply（每批 50，失败可续）

③ 语义定位：「上周那份转专业的通知标题太长了，改短点」
   search_knowledge("转专业 通知") → 候选 3 份 → 让用户选（不猜） → 进入 ①
```

链路 ② 是新增的，也是最容易出事的：**必须先 dry-run 出「将影响 N 条」再执行**，而不是直接改。批量改 500 条时，中间的失败要有断点续跑，不能一错全废。

### 7.4 审计与回滚

变更单（现有 `admin_change`）已经是这个结构，扩展两点：

- **保留 undo 快照**：apply 前把原值整行存进变更单，支持"撤销上一次变更"。
- **批量变更单记录子项状态**：500 条里成功了 480 条，要能看出失败的 20 条是哪几条、为什么。

---

## 八、学生端登录

### 8.1 登录带来的不只是权限

```
学号 → 学生画像（学院/专业/年级/GPA/兴趣）→ 自动进入会话上下文
```

这解决了上一版方案里最别扭的地方：现在问「我适合参加什么竞赛」，系统不知道你是大几，只能反问。登录后**画像就是隐式上下文**，planning 类问题少一轮交互。

### 8.2 实现取舍

| 方案 | 适用 | 代价 |
|---|---|---|
| JWT（无状态） | 演示/单校 | 无法主动踢人，需要短 TTL + refresh |
| 服务端 session（Redis） | 需要强制下线 | 多一次 Redis 往返 |
| 对接学校 CAS/OAuth | 真实上线 | 依赖学校，开发期难调试 |

**建议**：开发期 JWT（`student_no` + 签名 + 7 天），上线接 CAS。切换时只改 `/api/auth/login` 一个接口的实现，其余代码拿到的是统一的 `current_student_no`。

### 8.3 一条要注意的副作用

登录后自动按学生学院过滤检索，看起来贴心，实际会出事：全校性文件、跨学院竞赛会被过滤掉。

现有代码其实已经处理了这个问题 —— `build_filter()` 里 `college` 用 `$in [目标学院, 全校]`。**这个设计在加登录后要保住，不能改成精确匹配。**

---

## 九、可行性分析

### 9.1 结论

| 维度 | 结论 | 依据 |
|---|---|---|
| 1000 在线 / 20–100 QPS | **单机 4 worker + async 可撑** | 在途 300，协程开销可忽略 |
| 1 万份 / 10 万 chunk | **单机 Qdrant 够用**（约 1 GB 内存） | §1.3 推导 |
| 秒级实时更新 | **可达**，前提是向量库服务化 + outbox | §6 |
| 异构解析 | **可达**，但 OCR 要按需 | §5.4 的 31 核·小时 |
| 管理端对话调整 | **已有雏形**，缺批量和 dry-run | §7 |

**真正的天花板不是数据库，是 LLM 配额。** 100 QPS 全走生成不可能；靠缓存 + 降级把"必须 LLM"的请求压到 40% 以下，才能在配额内跑起来。

### 9.2 升级阈值（超过这些才需要考虑下一层架构）

| 指标 | 阈值 | 到时做什么 |
|---|---|---|
| 持续 QPS | > 200 | 应用层加机器 + Redis 集中缓存 |
| chunk 总数 | > 100 万 | Qdrant 分片 / 独立节点 |
| 单表行数 | > 1 亿 | 才考虑分库分表 |
| LLM 日调用 | 接近配额上限 | 提高缓存命中率 / 换更便宜的模型做粗筛 |

这些阈值是经验值，必须用压测校准 —— **压测之前不要按这些数字做架构决策**。

### 9.3 分阶段落地

| 阶段 | 内容 | 前置 | 验证方式 |
|---|---|---|---|
| **S0** | 全链路 async、连接池调参、缓存 L1 | 无 | 压测：并发 100 时无排队 |
| **S1** | Chroma → Qdrant、向量库服务化 | S0 | 多 worker 下新文件立即可搜 |
| **S2** | ingest_task + Worker 池、异步入库 | S1 | 批量导 1000 份不掉请求 |
| **S3** | outbox + kb_version、秒级可见 | S1 | 改一份文件，学生端 3s 内看到 |
| **S4** | 学生登录 + 画像自动注入 | 无（可并行） | planning 类不再反问年级 |
| **S5** | 管理端批量变更 + dry-run + undo | S3 | 批量改 100 条可回滚 |

S0 和 S4 互不依赖，可以并行；S1 是所有实时性工作的地基，必须先做。

---

## 十、明确不做的事

1. **不分库分表** —— 1 万学生、1 万文件、10 万 chunk，单库绰绰有余。
2. **不引入消息队列（Kafka/RabbitMQ）** —— 用 MySQL 任务表 + outbox 已经够；多一个中间件就多一套运维。
3. **不做全量 OCR** —— 按需 + 低优先级队列，先保证 90% 可用。
4. **不做 WebSocket 推送更新** —— 秒级可见已经满足需求。
5. **不让模型改正文** —— 只能发新版本，保住证据链。
6. **不做语义缓存** —— 实现复杂、命中判定难解释，先用精确 + 归一化。

---

## 附录：改动落点

| 目标 | 现有文件 | 改动 |
|---|---|---|
| 全链路 async | `app/api/main.py`、`app/agent/graph.py` | `def` → `async def`，`invoke` → `ainvoke` |
| 连接池 | `app/db.py:35` | 加 `pool_size/max_overflow` |
| 向量库换 Qdrant | `app/rag/vectorstore.py`（175 行，6 个函数） | **只改这一个文件**，接口不变 |
| 异步入库 | `app/ingest.py` | 拆成 `parse()` / `index()` 两段，供 Worker 调用 |
| outbox | 新增 `app/index_sync.py` | 事务写 + 消费循环 |
| 学生登录 | 新增 `app/api/student_auth.py` | JWT + `current_student_no` 依赖 |
| 批量变更 | `app/admin_change.py` | 支持 `batch_id` + 子项状态 |
