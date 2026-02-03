# Hybrid RAG Architecture Proposal: Vector-First Routing

> 이 문서는 2026-01-29 논의된 Vector RAG와 PageIndex RAG의 하이브리드 구성 방안을 정리한 제안서입니다. 추후 시스템 고도화 시 참조합니다.

---

## 1. 배경 및 문제점

### 현재 PageIndex Router의 한계
1. **파일명 의존성**: Router가 문서의 파일명(텍스트)만 보고 관련성을 판단하므로, 키워드가 포함되지 않은 관련 문서를 누락(Recall 저하)할 가능성이 높음.
2. **정보 부족**: 문서의 구체적인 내용(Semantic Context)을 모른 채 라우팅을 수행함.

### PageIndex 단독 검색의 한계
1. **속도 이슈**: 모든 문서에 대해 트리를 로드하고 검색하는 것은 비효율적임.
2. **Title 의존**: Section 검색 시 제목(Title)에 의존하므로, 제목에 드러나지 않은 세부 내용을 놓칠 수 있음.

---

## 2. 제안: Vector-First Routing (Hybrid Approach)

Vector Search의 장점(빠른 속도, 의미 기반 검색)을 활용하여 **"정확한 문서 선별(Routing)"** 을 수행하고, PageIndex의 장점(구조적 이해, 맥락 파악)으로 **"정밀 답변"** 을 생성하는 2단계 파이프라인입니다.

### 🏛️ 아키텍처

#### Stage 1: Macro Search (Candidate Selection)
- **도구**: Vector Search (기존 VectorStore 활용)
- **대상**: 전체 문서 (All Documents)
- **방법**: User Query로 Top-k 청크 검색 (예: k=10)
- **목적**: 정답이 있을 가능성이 높은 **"후보 문서군(Source Documents)"** 식별
- **로직**:
    1. Query -> Vector Search -> 10개 청크 반환
    2. 반환된 청크들의 `source` (파일명)를 집계 (Aggregation)
    3. 가장 많이 등장하거나 스코어가 높은 **상위 2~3개 문서**를 타겟으로 선정

#### Stage 2: Micro Search (Context Retrieval)
- **도구**: PageIndex Search
- **대상**: Stage 1에서 선정된 2~3개 문서
- **방법**: 선정된 문서의 트리 구조 로드 -> Query에 대한 관련 섹션 탐색
- **목적**: 문서의 계층적 구조와 긴 맥락을 파악하여 Vector가 놓친 주변 정보 확보

#### Stage 3: Answer Generation
- **입력**: `Vector Chunks` (구체적 사실) + `PageIndex Sections` (구조적 맥락)
- **모델**: HCX-007 (Reasoning/Thinking Mode)
- **기대 효과**:
    - **디테일**: Vector가 찾아낸 구체적 수치, 날짜 등 반영
    - **구조**: PageIndex가 찾아낸 조항의 범위, 예외 규정 등 반영

---

## 3. 구현 로직 (Pseudocode)

```python
class HybridRAG:
    def __init__(self):
        self.vector_rag = VectorRAG()
        self.pageindex_rag = PageIndexRAG()

    def search(self, query: str) -> List[Dict]:
        # 1. Vector Search로 후보 문서 식별 (Routing)
        vector_results = self.vector_rag.search(query, top_k=10)
        
        # 문서별 관련도 스코어링
        doc_scores = {}
        for res in vector_results:
            doc_name = res['metadata']['source']
            score = res['score']
            doc_scores[doc_name] = doc_scores.get(doc_name, 0) + score
            
        # Top-2 문서 선정
        target_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[:2]
        print(f"Refined Targets by Vector: {target_docs}")

        # 2. 선정된 문서에 대해 PageIndex 검색 수행
        pageindex_results = []
        for doc in target_docs:
            results = self.pageindex_rag.search(doc_path, query)
            pageindex_results.extend(results)

        # 3. 결과 병합 (Duplicate 제거 등)
        final_context = merge_results(vector_results, pageindex_results)
        return final_context
```

## 4. 기대 효과

| 구분 | 기존 Router | **Hybrid Routing** |
| :--- | :--- | :--- |
| **정확도 (Recall)** | 낮음 (파일명 매칭) | **높음 (Semantic 매칭)** |
| **효율성** | 중간 (LLM 호출 비용) | **높음 (Vector 연산은 저비용)** |
| **답변 품질** | 단일 소스 의존 | **상호 보완 (Detail + Structure)** |

---

## 5. 향후 과제

1. **Vector Index 재활용**: PageIndex 시스템 내에 Vector Store 통합
2. **Reranking**: 병합된 결과의 우선순위 재산정 (Reranker 모델 도입 고려)

---

## 6. 아키텍처 비교 분석 (Vector 실행 시점)

### Option A: Vector First (Pre-filtering) - **Recommended**
- **개념**: Vector Search를 필터(깔때기)로 사용하여 탐색할 문서를 먼저 좁히는 방식.
- **장점**:
    - **효율성**: PageIndex(LLM)는 비용이 높고 느리므로, 실행 대상을 줄이는 것이 필수적임.
    - **노이즈 제거**: 관련 없는 문서를 미리 배제하여 할루시네이션 방지.
- **단점**: Vector 단계에서 놓친 문서는 복구 불가능 (Cascading Error).
- **보완책**: Vector 검색 범위를 넓게(Top-10~20) 설정하여 Recall 확보.

### Option B: Parallel / Vector Last (Ensemble)
- **개념**: 두 방식을 독립적으로 병렬 실행하거나, PageIndex 후 Vector로 보완하는 방식.
- **장점**: 상호 보완을 통한 최대 재현율(Recall) 확보.
- **단점**:
    - **속도/비용**: 어떤 문서에 대해 PageIndex를 돌릴지 모르는 상태라면 많은 문서를 스캔해야 하므로 **실시간 시스템에 부적합**.
    - **복잡성**: 서로 다른 성격의 결과를 병합(Rank Fusion)하는 난이도가 놂음.

### 결론
실시간 응답 속도와 비용 효율성이 중요한 본 시스템에서는 **Option A (Vector First)** 방식이 현실적인 최선입니다.
