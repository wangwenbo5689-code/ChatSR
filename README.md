<h1 align="center">ChatPDF</h1>
<div align="center">
  <a href="https://github.com/shibing624/ChatPDF">
  </a>

<p align="center">
    <h3>基于本地 LLM 做检索知识问答(RAG)</h3>
    <p align="center">
      <a href="https://github.com/shibing624/ChatPDF/blob/main/LICENSE">
        <img alt="Tests Passing" src="https://img.shields.io/github/license/shibing624/ChatPDF" />
      </a>
      <a href="https://gradio.app/">
        <img alt="GitHub Contributors" src="https://img.shields.io/badge/Base-Gradio-fb7d1a?style=flat" />
      </a>
      <p>
        根据文件回答 / 开源模型 / 本地部署LLM
      </p>
    </p>
    <p align="center">
      <img alt="Animation Demo" src="https://github.com/shibing624/ChatPDF/blob/main/docs/snap.png" width="860" />
    </p>
  </p>
</div>


## 介绍
- 本项目实现了轻量版的GraphRAG
  - 支持`local`模式的关系图检索的文档问答
  - 支持Openai API, Deepseek API, Ollama API等，可自行扩展支持更多LLM
  - 支持openai embedding、本地 text2vec embedding、huggingface embedding、sentence-transformers embedding等
  - 异步开发，支持多个API并发请求
- 本项目支持多种开源LLM模型，包括Qwen/DeepSeek等
- 本项目支持多种文件格式，包括PDF、docx、txt等
- 本项目优化了RAG准确率
  - Chinese chunk切分优化，适配中英文混合文档
  - embedding优化，使用text2vec的sentence embedding，支持sentence embedding/字面相似度匹配算法
  - 检索匹配优化，引入jieba分词的rank_BM25，提升对query关键词的字面匹配，使用字面相似度+sentence embedding向量相似度加权获取corpus候选集
  - 新增reranker模块，对字面+语义检索的候选集进行rerank排序，减少候选集，并提升候选命中准确率，用`rerank_model_name_or_path`参数设置rerank模型
  - 新增候选chunk扩展上下文功能，用`num_expand_context_chunk`参数设置命中的候选chunk扩展上下文窗口大小
  - RAG底模优化，可以使用200k的基于RAG微调的LLM模型，支持自定义RAG模型，用`generate_model_name_or_path`参数设置底模
- 本项目基于gradio开发了RAG对话页面，支持流式对话

## 原理

<img src="https://github.com/shibing624/ChatPDF/blob/main/docs/chatpdf.jpg" width="860" />

## Usage

### 安装依赖

在终端中输入下面的命令，然后回车即可。
```shell
pip install -r requirements.txt
```

如果您在使用Windows，建议通过WSL，在Linux上安装。如果您没有安装CUDA，并且不想只用CPU跑大模型，请先安装CUDA。

如果下载慢，建议配置豆瓣源。

### RAG示例

请使用下面的命令。取决于你的系统，你可能需要用python或者python3命令。请确保你已经安装了Python。
```shell
CUDA_VISIBLE_DEVICES=0 python rag.py
```

output:

```
prompt: 基于以下已知信息，用专业知识回答用户的问题。用简体中文回答。

已知内容:
[1]	 "ReferencesPeter F Brown, John Cocke, Stephen A Della Pietra, Vincent J Della Pietra, Fredrick Jelinek, John DLafferty, Robert L Mercer, and Paul S Roossin. A statistical 
[2]	 "Let be an encoder that infers the content zfor a given sentence xand a styley
...

问题:
自然语言中的非平行迁移是指什么？

---
回答:
自然语言中的非平行迁移是指在文本生成任务中，我们只能假设访问到非平行或单语的文本数据。这类任务包括翻译和摘要，其中所有问题都涉及这类任务。 
['[1]\t "ReferencesPeter F Brown, John Cocke, Stephen A Della Pietra, Vincent J Della Pietra, Fredrick Jelinek, John DLafferty, Robert L Mercer, and Paul S Roossin. A statistical approach to machine translation.Computational linguistics 
'[2]\t "LetE:X\x02Y!Z be an encoder that infers the content zfor a given sentence xand a styley, andG:Y\x02Z!X be a generator that generates a sentence xfrom a given style yand contentz.EandGform an auto-encoder ', 
...
]
```

### 启动 Gradio 的 Web 服务

#### 使用默认的 Ollama qwen2.5:1.5b 模型（推荐）

首先确保你已经安装并运行了 Ollama，并且拉取了 qwen2.5:1.5b 模型：

```bash
# 如果没有安装 Ollama，请访问 https://ollama.ai 下载安装
# 拉取模型
ollama pull qwen2.5:1.5b

# 启动 Ollama 服务（如果还未启动）
ollama serve
```

然后运行 ChatPDF：

```shell
python webui.py
```

#### 使用其他模型配置

如果需要指定 Ollama 主机地址或模型名称：

```shell
python webui.py --gen_model_type ollama --gen_model_name qwen2.5:1.5b --ollama_host http://127.0.0.1:11434
```

#### 使用本地 Transformers 模型（可选）

如果需要使用本地的 Qwen/Qwen2-0.5B-Instruct 模型：

```shell
python webui.py --gen_model_type auto --gen_model_name Qwen/Qwen2-0.5B-Instruct
```

现在，你应该已经可以在浏览器地址栏中输入 http://localhost:8082 查看并使用 ChatPDF 了。

### GraphRAG 示例
> [!TIP]
>
>  **Please set OpenAI API key in environment: `export OPENAI_API_KEY="sk-..."`.** 
>

## Ollama 本地模型快速开始

### 1. 安装 Ollama

访问 [Ollama 官网](https://ollama.ai) 下载并安装适合你系统的 Ollama。

### 2. 拉取模型

```bash
# 拉取默认的 qwen2.5:1.5b 模型
ollama pull qwen2.5:1.5b

# 或者拉取其他模型（可选）
ollama pull qwen2:7b
ollama pull llama3
```

### 3. 启动 Ollama 服务

```bash
# 在后台启动 Ollama 服务
ollama serve
```

### 4. 验证安装

```bash
# 测试模型是否正常工作
ollama run qwen2.5:1.5b "你好"
```

### 5. 运行 ChatPDF

```bash
# 使用默认的 Ollama 模型
python webui.py

# 或者指定模型和主机
python webui.py --gen_model_type ollama --gen_model_name qwen2.5:1.5b --ollama_host http://127.0.0.1:11434
```

### 6. 测试配置

```bash
# 运行配置测试
python rag.py --test
```

如果测试通过，你就可以在浏览器中访问 http://localhost:8082 开始使用了！

## 常见问题

### Q: 如何切换不同的模型？
A: 修改 `--gen_model_name` 参数即可，例如：
```bash
python webui.py --gen_model_name qwen2:7b
```

### Q: Ollama 服务不在默认端口？
A: 使用 `--ollama_host` 参数指定正确的主机和端口：
```bash
python webui.py --ollama_host http://your-server:11434
```

### Q: 想改回使用本地的 Qwen/Qwen2-0.5B-Instruct 模型？
A: 修改模型类型和名称：
```bash
python webui.py --gen_model_type auto --gen_model_name Qwen/Qwen2-0.5B-Instruct
```

## Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/ChatPDF.svg)](https://github.com/shibing624/ChatPDF/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/ChatPDF/blob/main/docs/wechat.jpeg" width="200" />

## License


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加ChatPDF的链接和授权协议。


## Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目。

### 关联项目推荐
- [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)：训练自己的GPT大模型，实现了包括增量预训练、有监督微调、RLHF(奖励建模、强化学习训练)和DPO(直接偏好优化)

