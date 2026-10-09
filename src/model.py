import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 镜像
os.environ["HF_HOME"] = "D:/claude-dev/rag-agent-training"           # 缓存目录

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)                    # ← 运行到这行，开始下载分词器
model = AutoModelForSequenceClassification.from_pretrained(model_name)   # ← 运行到这行，开始下载模型