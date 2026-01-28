# PageIndex with HyperCLOVA X Reasoning Model

**Comparative RAG Experiment using NCloud HCX-007 (HyperCLOVA X Reasoning Model)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[한국어 버전](README.md)

---

## 📌 Project Overview

This project applies the structure-based document retrieval approach from the [PageIndex](https://github.com/ofirpress/pageindex) paper to **NCloud HyperCLOVA X (HCX-007)** reasoning model, comparing its performance against traditional Vector RAG methods.

### 🎯 Objectives

1. **Validate PageIndex applicability to Korean legal documents**
2. **Evaluate HCX-007 as an OpenAI alternative for RAG systems**
3. **Quantitative comparison: Vector RAG vs Tree-based RAG**

---

## 🧪 Experiment Results Summary

### Evaluation Dataset
- **Golden Set**: 20 QA pairs
  - AI Basic Act & 5 Guidelines: 15 questions
  - National Core Technology Cloud Security Guidelines: 5 questions

### Performance Comparison (N=20)

| Metric | Vector RAG | PageIndex RAG |
|---|:---:|:---:|
| **Avg Score (1-5)** | **3.15** | 2.85 |
| **Avg Response Time** | **17.5s** | 107.3s |
| **Document Retrieval Hit Rate** | 15% | **25%** |

> *Excluding 0-score error cases: Vector RAG 3.47 vs PageIndex RAG 3.0*

### Key Findings

| Finding | Description |
|---|---|
| ✅ **Vector RAG Strengths** | Fast response, higher answer quality |
| ✅ **PageIndex Strengths** | Accurate document selection (Global Routing) |
| ⚠️ **HCX-007 Limitations** | Inconsistent JSON output, slower response times |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Comparison UI (Gradio)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐     ┌─────────────────────────┐   │
│  │     Vector RAG      │     │     PageIndex RAG       │   │
│  ├─────────────────────┤     ├─────────────────────────┤   │
│  │ • Semantic Chunking │     │ • Tree Structure        │   │
│  │ • Qdrant Vector DB  │     │ • Global Router         │   │
│  │ • Top-K Retrieval   │     │ • LLM-based Navigation  │   │
│  └─────────────────────┘     └─────────────────────────┘   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  NCloud HCX-007 (LLM Backend)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
PageIndex/
├── comparison/                 # Comparison experiment modules
│   ├── modules/               # RAG implementations
│   │   ├── vector_rag.py      # Vector RAG
│   │   ├── pageindex_rag.py   # PageIndex RAG
│   │   ├── pageindex_router.py # Global Routing
│   │   └── ncloud_llm.py      # HCX-007 Wrapper
│   ├── data/
│   │   ├── documents/         # Test PDF documents
│   │   ├── cache/             # Tree cache
│   │   └── results/           # Evaluation results
│   └── evaluator.py           # Automated evaluation script
├── comparison_ui.py           # Gradio UI
├── pageindex/                 # Original PageIndex library
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API Key Configuration

Create `.env` file:
```env
NCLOUD_API_KEY=your_api_key
NCLOUD_API_URL=https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007
```

### 3. Run UI

```bash
python comparison_ui.py
# Open http://127.0.0.1:7860 in browser
```

### 4. Run Evaluation

```bash
python comparison/evaluator.py
# Results: comparison/data/results/
```

---

## 📚 Lessons Learned

### 1. HCX-007 Limitations as OpenAI Replacement

| Issue | Cause | Solution |
|---|---|---|
| **Frequent JSON parsing errors** | Reasoning process included in response | Added Regex fallback |
| **6x slower response time** | `thinking_effort` parameter impact | Adjust to Low/Medium |
| **Unstable structured output** | Tendency to ignore prompt constraints | Enhanced post-processing |

### 2. PageIndex Characteristics

- **Pros**: Effective for well-structured legal/technical documents
- **Cons**: Multiple LLM calls required for tree generation → increased cost/time
- **Improvement**: Caching strategy can offset initial costs

### 3. Evaluation Methodology

- **LLM-as-a-Judge** approach heavily depends on JSON output consistency
- Robust parsing logic (Regex, bracket matching) is essential

---

## 🔮 Future Improvements

1. **Explore HCX-007 JSON mode**: Investigate structured output API options
2. **Improve Global Routing accuracy**: Enhanced prompt engineering
3. **Hybrid approach**: Experiment with Vector + PageIndex combination
4. **Large-scale evaluation**: 100+ questions for statistical significance

---

## 📄 References

- [PageIndex Paper](https://arxiv.org/abs/2501.xyz) - Original research
- [NCloud HyperCLOVA X](https://www.ncloud.com/product/aiService/clovaStudio) - LLM API
- [Qdrant](https://qdrant.tech/) - Vector Database

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

---

*Last Updated: 2026-01-28*
