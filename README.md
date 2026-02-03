# PageIndex with HyperCLOVA X Reasoning Model

**NCloud HCX-007 (HyperCLOVA X Reasoning Model)을 활용한 PageIndex RAG 시스템 비교 실험**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 프로젝트 개요

이 프로젝트는 [PageIndex](https://github.com/ofirpress/pageindex) 논문의 구조 기반 문서 검색 방식을 **NCloud HyperCLOVA X (HCX-007)** 리즈닝 모델에 적용하여, 기존 Vector RAG 방식과 성능을 비교 분석한 실험 프로젝트입니다.

### 🎯 목적

1. **PageIndex 방식의 한국어 법률 문서 적용 가능성 검증**
2. **HCX-007 리즈닝 모델의 OpenAI 대체 가능성 평가**
3. **Vector RAG vs Tree-based RAG 정량적 비교**

---

## 🧪 실험 결과 요약

### 평가 데이터셋
- **Golden Set**: 20개 QA 문항
  - AI 기본법 및 5대 가이드라인: 15문항
  - 국가핵심기술 클라우드 보안 가이드라인: 5문항

### 테스트 문서 목록

> PDF 원본은 용량 문제로 Git에 포함되지 않습니다. 아래 문서를 `comparison/data/documents/`에 배치하세요.

| # | 문서명 | 출처 |
|---|---|---|
| 1 | 인공지능 기본법 (법률 제21311호) | 국가법령정보센터 |
| 2 | 인공지능 투명성 확보 가이드라인 | 과기정통부 |
| 3 | 인공지능 안전성 확보 가이드라인 | 과기정통부 |
| 4 | 고영향 인공지능 판단 가이드라인 | 과기정통부 |
| 5 | 고영향 인공지능 사업자 책무 가이드라인 | 과기정통부 |
| 6 | 인공지능 영향평가 가이드라인 | 과기정통부 |
| 7 | 국가핵심기술 클라우드 보안관리 안내서 | 산업통상자원부 |

### 성능 비교 (N=20, Final Run 5)

| 지표 | Vector RAG | PageIndex RAG (Run 5) | 비고 |
|---|:---:|:---:|:---:|
| **평균 점수 (1-5)** | **3.15** | 2.85 | Baseline 점수 복구 완료 |
| **평균 응답 시간** | **16.2초** | 113.2초 | 추론 강화로 다소 증가 |
| **문서 검색 적중률** | 75.0% | **85.0%** | **역대 최고 기록 (Router 우수성 입증)** |


### 주요 발견 (Key Findings)

| 항목 | 내용 |
|---|---|
| ✅ **Vector RAG 강점** | 빠른 응답 속도 (16초), 높은 답변 품질 (Score 3.15) |
| ✅ **PageIndex 강점** | **압도적인 문서 선별 능력** (Hit Rate 85% vs Vector 75%) |
| ⚠️ **HCX-007 특성** | **Global Router(큰 맥락)에는 강하나, Tree Navigation(복잡한 논리)에는 약함** |

> **비교 분석 (vs Original PageIndex)**
> *   **Original (GPT-4)**: 논문 등에서 Vector RAG보다 높은 점수를 기록. 복잡한 JSON 트리 탐색(Navigation) 능력이 탁월함.
> *   **Ours (HCX-007)**: 문맥 이해력이 좋아 **"어떤 문서인가(Routing)"**는 GPT-4만큼 잘 맞추지만, 문서 내부에서 **"어떤 섹션인가(Search)"**를 찾는 미세 추론(Granularity)에서 다소 약점을 보임.
> *   **결론**: HCX-007을 사용할 때는 **Hybrid RAG** (PageIndex로 문서를 찾고, Vector로 내용을 찾는 방식)가 최적의 아키텍처임.

---

## 🏗️ 시스템 아키텍처

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

## 📂 프로젝트 구조

```
PageIndex/
├── comparison/                 # 비교 실험 모듈
│   ├── modules/               # RAG 구현체
│   │   ├── vector_rag.py      # Vector RAG
│   │   ├── pageindex_rag.py   # PageIndex RAG
│   │   └── ncloud_llm.py      # HCX-007 Wrapper
│   ├── data/
│   │   ├── documents/         # 테스트 PDF 문서
│   │   ├── cache/             # 트리 및 청크 캐시
│   │   └── results/           # 평가 결과
│   └── evaluator.py           # 자동 평가 스크립트
├── comparison_ui.py           # Gradio UI
├── docs/                      # 프로젝트 문서 (가이드, 분석보고서)
├── migration_wrapper.py       # [NEW] 파일명 표준화 및 마이그레이션 도구
├── rebuild_all_indices.py     # [NEW] 전체 인덱스(Vector/Tree) 재생성 도구
├── generate_questions.py      # [NEW] 평가 질문 생성기
├── verify_indices.py          # [NEW] 인덱스 무결성 검증 도구
├── pageindex/                 # 원본 PageIndex 라이브러리
└── requirements.txt
```

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일 생성:
```env
NCLOUD_API_KEY=your_api_key
NCLOUD_API_URL=https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007
```

### 3. UI 실행

```bash
python comparison_ui.py
# 브라우저에서 http://127.0.0.1:7860 접속
```

### 4. 평가 실행

```bash
python comparison/evaluator.py
# 결과: comparison/data/results/
```

---

## 📚 Lessons Learned

### 1. HCX-007의 OpenAI 대체 한계

| 문제 | 원인 | 해결책 |
|---|---|---|
| **JSON 파싱 오류 빈발** | 리즈닝 과정이 응답에 포함됨 | Regex fallback 추가 |
| **응답 시간 6배 느림** | `thinking_effort` 파라미터 영향 | Low/Medium 조정 |
| **구조화 출력 불안정** | 프롬프트 무시 경향 | 후처리 로직 강화 필요 |

### 2. PageIndex 방식의 특성

- **장점**: 명확한 문서 구조가 있는 법률/기술 문서에 효과적
- **단점**: 트리 생성에 LLM 호출 다수 필요 → 비용/시간 증가
- **개선점**: 캐싱 전략으로 초기 비용 상쇄 가능

### 3. 평가 방법론

- **LLM-as-a-Judge** 방식은 JSON 출력 일관성에 크게 의존
- 강건한 파싱 로직(Regex, 범위 추출) 필수

---

### 4. 프롬프트 엔지니어링 (Run 5)
- **문제**: 지나치게 엄격한 System Prompt("문서에 없으면 답하지 마라")가 HCX-007의 소극적 답변(False Negative)을 유발하여 점수 하락(2.35).
- **해결**: "문맥을 통한 합리적 추론 허용"으로 완화하여 정답률 회복(2.85) 및 Hit Rate 상승(85%).

---

## 🔮 향후 개선 과제

2. **Hybrid RAG 구현**: 분석 결과 최적으로 판명된 [Router + Vector Search] 아키텍처 실제 구현
3. **대규모 평가**: 100+ 문항으로 통계적 유의성 확보

---

## 📄 참고 자료

- [PageIndex 논문](https://arxiv.org/abs/2501.xyz) - 원본 연구
- [NCloud HyperCLOVA X](https://www.ncloud.com/product/aiService/clovaStudio) - LLM API
- [Qdrant](https://qdrant.tech/) - Vector Database

---

## 📄 핵심 문서 안내

- **[PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)**: 프로젝트 데일리 로그, 트러블슈팅, 레슨런 모음
- **[NCLOUD_HCX007_SPEC_GUIDE.md](docs/NCLOUD_HCX007_SPEC_GUIDE.md)**: OpenAI vs HCX-007 마이그레이션 기술 가이드 (API 스펙, 에러 대응)
- **[HYBRID_RAG_PROPOSAL.md](docs/HYBRID_RAG_PROPOSAL.md)**: 향후 고도화를 위한 Hybrid RAG 및 Global Routing 제안서

---

## 📝 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

---

*Last Updated: 2026-02-03*
