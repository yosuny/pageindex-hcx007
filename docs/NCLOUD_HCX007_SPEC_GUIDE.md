# NCloud HCX-007 API Technical Guide & Lessons Learned

이 문서는 프로젝트 진행 중 OpenAI API에서 NCloud CLOVA Studio(HCX-007)로 전환하며 겪은 주요 스펙 차이, 기술적 이슈, 그리고 모델의 특성을 정리한 가이드입니다.

---

## 1. API Spec Differences (OpenAI vs. NCloud)

| 항목 | OpenAI API | NCloud HCX-007 (CLOVA Studio) |
| :--- | :--- | :--- |
| **Endpoint** | `https://api.openai.com/v1/chat/completions` | `https://clovastudio.stream.ntruss.com/testapp/v1/...` (Test App)<br>또는 APIGW 경로 |
| **Auth Header** | `Authorization: Bearer $KEY` | `X-NCP-CLOVASTUDIO-API-KEY: $KEY` (또는 GW용 $AUTH) |
| **Custom Header** | N/A | `X-NCP-CLOVASTUDIO-REQUEST-ID`, `X-NCP-APIGW-API-KEY` |
| **Model Params** | `model`, `temperature`, `max_tokens` | `api_key`가 URL/Header에 포함, **`thinking_effort`** 추가 |
| **Response Format** | `response_format: { "type": "json_object" }` | 공식 지원 중이나, Thinking Mode와 혼합 시 파싱 주의 필요 |

---

## 2. 주요 기술 이슈 및 해결책 (Issues & Solutions)

### 2.1. 401 Unauthorized (인증 오류)
- **증상**: API 호출 시 401 에러 발생.
- **원인**: CLOVA Studio 직접 호출 시와 API Gateway(APIGW)를 거칠 때의 헤더 스펙이 다름. APIGW 사용 시 `X-NCP-APIGW-API-KEY`와 별도의 `Authorization` 헤더 조합이 필수적임.
- **해결**: `ncloud_llm.py`에서 두 종류의 키(`API_KEY`, `APIGW_API_KEY`)를 모두 처리하도록 래퍼 고도화.

### 2.2. Thinking Mode와 JSON 파싱 충돌
- **증상**: `thinking_effort` 활성화 시, 응답에 `<thought>...</thought>` 태그나 리즈닝 과정이 포함되어 `json.loads()`가 실패함.
- **원인**: 모델이 답변을 내놓기 전 사고 과정을 텍스트로 함께 출력함.
- **해결**: **Regex Fallback** 로직 도입.
    - 1차: `json.loads()` 시도.
    - 2차: 정규표현식(`r'\{.*\}'`)으로 최외곽 중괄호 영역만 추출 후 파싱.
    - 3차: 특정 필드(예: `score`)만 개별 추출하는 방어적 파싱 적용.

### 2.3. 429 Too Many Requests (Rate Limit)
- **증상**: 전처리 또는 평가 시 대량 요청 발생 시 API 차단.
### 2.4. Streaming Response Protocol 차이
- **증상**: OpenAI 스타일로 `chunk.choices[0].delta.content`를 기대하면 파싱 실패.
- **원인**: NCloud는 SSE(Server-Sent Events)에서 고유한 이벤트 타입(`event: result` vs `event: token`)을 사용함.
    - OpenAI: 토큰 단위로 계속 스트리밍.
    - NCloud: 토큰도 오지만, 마지막에 `event: result`에 **완성된 전체 텍스트**가 담겨 옴.
- **해결**: `ncloud_llm.py`에서 `event: result`를 감지하여 최종 응답으로 처리하는 로직 구현. 중간 토큰 수집보다 `result` 이벤트 처리가 더 안정적임.

### 2.5. Parameter Naming (Snake_case vs CamelCase)
- **차이점**: OpenAI SDK는 Pythonic한 `snake_case`를 쓰지만, HCX-007 Raw API는 `camelCase`를 요구함.
    - `max_tokens` -> **`maxCompletionTokens`**
    - `frequency_penalty` -> **`repetitionPenalty`**
    - `stop` -> **`stopBefore`**
- **조치**: Wrapper 클래스(`NCloudLLM`) 내부에서 파라미터 매핑 딕셔너리를 두어 변환 처리.

### 2.6. Embedding API Format 차이
- **OpenAI**: Input이 `str` 또는 `List[str]`. Output은 `{ "data": [ { "embedding": [...] } ] }`.
- **NCloud**: Input이 `text` (String) 하나만 허용되는 경우가 많음(v2). Batch 처리 시 Loop 필요.
- **해결**: `ncloud_embedding.py`에서 `embed_documents(list)` 호출 시 내부적으로 Loop를 돌며 개별 호출 + 에러 핸들링(Retry) 구조로 변경.

### 2.7. Tokenizer Mismatch (Tiktoken vs HCX)
- **이슈**: 원본 PageIndex는 OpenAI의 `tiktoken`을 사용하여 컨텍스트 윈도우를 계산함. HCX-007은 독자적인 Tokenizer를 사용하므로 토큰 수 계산이 정확하지 않음.
- **영향**: `tiktoken` 기준으로는 리미트 내라고 판단했으나, 실제 HCX API 호출 시 `Text too long` 에러가 발생할 수 있음.
- **해결**: 안전 마진(Safety Margin)을 10~20% 더 확보하거나, 문자 수(Char count) 기반의 보수적인 청킹 전략을 병행해야 함.

### 2.8. Structured Output vs. Thinking Mode (Trade-off)
- **시도**: 안정적인 JSON 파싱을 위해 HCX의 `responseFormat: { "type": "json_object" }` 기능을 적용 시도했습니다.
- **제약**: 현재 API 스펙상 **Thinking Mode(`thinking_effort`)와 Structured Output은 동시에 사용할 수 없는(Mutually Exclusive) 관계**임이 확인되었습니다.
- **결정**: 법률 RAG 특성상 답변의 "논리적 정확성(Reasoning)"이 "형식적 파싱 용이성"보다 중요하다고 판단하여, **Structured Output을 포기하고 Thinking Mode를 선택**했습니다.
- **대안**: 대신 포맷 안정성은 `Regex Fallback` 로직(2.2절 참조)으로 보완했습니다.

---

## 3. HCX-007 프롬프팅 및 특징

### 3.1. 장점 (Pros)
- **한국어 전문성**: 법률, 공공 가이드라인 등 한국어 특유의 문맥과 한자어 이해도가 GPT-4o 대비 뛰어나며, 표현이 훨씬 자연스러움.
- **Thinking Mode (Reasoning)**: 질문이 복잡할수록(`low` -> `medium`) 답변의 논리적 구조가 탄탄해짐. 특히 "근거 조항 찾기"에서 강점을 보임.
- **데이터 프라이버시**: 국내 인프라를 사용하므로 공공/금융 민감 데이터 처리에 적합.

### 3.2. 단점 (Cons)
- **속도(Latency) 이슈 (Test App 한계)**: 현재 프로젝트는 **CLOVA Studio Test App** 환경을 사용 중입니다. 실제 서비스 앱 대비 TPS(초당 처리량) 제한이 엄격하고 응답 속도가 느릴 수 있음을 감안해야 합니다. (Thinking 모드의 연산 부하와 결합되어 더 느리게 체감됨)
- **출력 일관성**: JSON 모드 지정 시에도 가끔 생각 과정을 섞어서 출력하는 경우가 있어 후처리가 필수적임.

### 3.3. 프롬프트 엔지니어링 히스토리 & Best Practices

프로젝트 진행 과정에서 HCX-007의 성능을 극대화하기 위해 다음과 같은 프롬프트 개선 과정을 거쳤습니다.

#### 1. 언어 최적화 (English -> Korean)
*   **초기**: 영어 System Prompt 사용 ("You are a legal expert...").
*   **문제**: 모델이 한국어 질문에 대해서도 영어로 생각하거나, 번역투의 어색한 답변을 생성.
*   **개선**: **전면 한국어 프롬프트 교체** ("당신은 법률 문서 분석 전문가입니다...").
*   **결과**: 답변의 자연스러움과 법률 용어 구사력이 대폭 향상됨.

#### 2. 할루시네이션 통제 (Strict Context Mode)
*   **문제**: 검색 결과에 없는 내용을 일반 지식으로 채워서 답변하는 경향.
*   **개선**: 부정 명령어와 경고를 반복적으로 배치.
    ```markdown
    1. 반드시 아래 [검색된 문서]에 포함된 내용만 사용하여 답변하세요.
    2. 검색된 문서에 없는 정보는 절대 사용하지 마세요.
    3. 추측하거나 일반 지식을 사용하지 마세요.
    ```
*   **Effect**: `Thinking Mode`가 이 제약 조건을 먼저 "인지"하고 답변을 생성하므로 신뢰도 급상승.

#### 3. 출처 표기 강제 (Citation Enforcement)
*   **문제**: "문서에 따르면..." 정도로만 뭉뚱그려 답변.
*   **개선**: 구체적인 포맷을 예시와 함께 제시.
    ```markdown
    - 답변 규칙: 각 문장의 끝에 반드시 출처를 표기하세요.
    - 형식: `[[실제파일명.pdf]] (p.페이지번호)`
    - 예시: [[AI기본법.pdf]] (p.12)
    ```
*   **결과**: HCX-007이 파일명과 페이지 정보를 정확히 매핑하여 출력함.

#### 4. Thinking Mode 유도
*   **전략**: 단순한 "답변해" 대신 **"Step-by-Step으로 분석하고 답변해"**라는 지시가 필수적.
*   **적용**: Evaluator에서 `thinking_effort="medium"` 설정 시, 프롬프트에 논리적 단계를 명시하면(1. 분석 -> 2. 검증 -> 3. 결론) 모델이 이를 그대로 따라가며 사고 과정을 전개함.

### 3.4 PageIndex 구현 시 특이사항 (PageIndex Specifics)
- **Recursive Summarization**: PageIndex는 트리를 만들 때 "요약의 요약"을 반복합니다. HCX-007은 이 과정에서 **Thinking Mode**를 켜면 트리의 깊이가 깊어질수록("Deep Dive") 요약 품질이 좋아지는 경향이 있습니다.
- **Nested JSON Stability**: 단순 JSON보다 계층적(Nested) JSON 생성 시 괄호 닫기 오류가 더 잦습니다. `pageindex_rag.py`에서는 이를 방지하기 위해 **"JSON 형식을 엄격히 준수하라"**는 시스템 프롬프트를 매 단계 주입하고, 파싱 실패 시 재시도하는 로직이 필수적입니다.

---

## 4. 결론
HCX-007은 **"속도보다는 정확도와 한국어 맥락이 중요한 도메인"**(예: 법률 RAG)에 최적화된 모델입니다. 다만, OpenAI 환경에서 이동 시 **인증 헤더 체계**와 **비정형 응답(Thinking)에 대한 파싱 처리**가 가장 큰 기술적 허들입니다.
