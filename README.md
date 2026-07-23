# LLM-Twin

# LLM-Twin-Architecture-Pipeline

## Project Overview

This project is an end to end Machine Learning pipeline designed to build a personalized AI digital twin. Based on architectural patterns from the "LLM Engineer's Handbook", it features a modular design incorporating an ETL data ingestion pipeline, Supervised Fine Tuning (SFT) of open weight models, Advanced RAG (Retrieval Augmented Generation), and an automated evaluation gate using LLM as a judge.

## Technology Stack

* **Language:** Python 3.10+
* **Large Language Models:** Meta Llama 3 8B (Base), OpenAI GPT-4o-mini (Evaluator)
* **Machine Learning & NLP:** Hugging Face Transformers, PEFT (LoRA), TRL, SentenceTransformers, CrossEncoder
* **Vector Database:** Qdrant
* **Data Processing:** LangChain (Text Splitters)
* **Framework:** PyTorch

## Project Architecture

```text
├── data/                    
│   ├── raw/                   # Raw extracted personal data
│   └── processed/             # Instruction pairs for fine tuning
├── application/             
│   ├── etl.py                 # Data extraction and warehouse loading
│   ├── vectorize.py           # Text chunking and Qdrant embedding
│   └── rag.py                 # Advanced RAG with expansion and reranking
├── model/                   
│   └── fine_tune.py           # LoRA SFT script for persona adaptation
├── evaluation/              
│   └── production_gate.py     # Golden set testing and factual evaluation
├── .env                       # Environment variables (gitignored)
├── .gitignore                 # Version control exclusion list
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation






