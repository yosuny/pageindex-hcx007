"""
RAG 비교 테스트 UI v3

Vector RAG와 PageIndex RAG의 답변을 나란히 비교할 수 있는 Gradio 기반 UI입니다.
- 좌측 사이드바: 문서 선택
- 메인 영역: 질문 → 답변 → 비교 요약
"""
import os
import sys
import time
import json
import hashlib
from datetime import datetime
import gradio as gr
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.modules.vector_rag import VectorRAG
from comparison.modules.pageindex_rag import PageIndexRAG
from comparison.modules.pageindex_router import PageIndexRouter

# Initialize RAG systems
print("Initializing RAG systems...")
vector_rag = VectorRAG(chunking_strategy="semantic")
pageindex_rag = PageIndexRAG(thinking_effort="medium")
pageindex_router = PageIndexRouter(thinking_effort="medium")
print("RAG systems initialized!")

# Logging & Caching Setup
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison", "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(LOG_DIR, "query_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def get_query_hash(question, docs):
    """Generate MD5 hash based on question and selected documents."""
    content = f"{question}|{'|'.join(sorted(docs))}"
    return hashlib.md5(content.encode()).hexdigest()

def get_recent_queries():
    """Get list of unique recent queries from history."""
    history = load_history()
    # Sort by timestamp desc
    sorted_items = sorted(history.values(), key=lambda x: x.get("timestamp", ""), reverse=True)
    # Extract unique queries
    queries = []
    seen = set()
    for item in sorted_items:
        q = item.get("query", "")
        if q and q not in seen:
            queries.append(q)
            seen.add(q)
    return queries[:15]  # Top 15 recent queries


def load_cached_result(query):
    """Load cached result for the selected query from history."""
    if not query:
        return "", "", "", gr.Dropdown(choices=get_recent_queries())
        
    history = load_history()
    # Find latest entry with this query
    matches = [h for h in history.values() if h.get("query") == query]
    
    if matches:
        # Sort by timestamp desc to get the latest
        matches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        cached = matches[0]
        
        # Format cached responses
        v_res = cached["vector_result"]
        p_res = cached["pageindex_result"]
        
        vector_output = f"**⏱️ {v_res.get('time', 0):.2f}초 (History)**\n\n{v_res.get('answer', '')}"
        
        pi_docs = p_res.get("docs_searched", 0)
        pageindex_output = f"**⏱️ {p_res.get('time', 0):.2f}초 (History)** ({pi_docs}개 문서 검색)\n\n{p_res.get('answer', '')}"
        
        comparison = f"""---
### 📊 비교 요약
| 항목 | Vector RAG | PageIndex RAG |
|:---|:---:|:---:|
| **응답 시간** | {v_res.get('time', 0):.2f}초 (History) | {p_res.get('time', 0):.2f}초 (History) |
| **답변 길이** | {len(v_res.get('answer', ''))} 자 | {len(p_res.get('answer', ''))} 자 |
"""
        return vector_output, pageindex_output, comparison, gr.Dropdown(choices=get_recent_queries())
    else:
        # No history found for this query
        msg = "⚠️ 저장된 결과가 없습니다. '비교 분석 실행' 버튼을 눌러주세요."
        return msg, msg, "", gr.Dropdown(choices=get_recent_queries())


# Get available PDFs
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison", "data", "documents")
pdf_files = []
if os.path.exists(PDF_DIR):
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])

# Detect existing PageIndex caches
pageindex_cached = set()
tree_cache_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "comparison", "data", "cache", "pageindex_trees"
)
if os.path.exists(tree_cache_dir):
    for cache_file in os.listdir(tree_cache_dir):
        if cache_file.endswith("_tree.json"):
            for pdf in pdf_files:
                pdf_stem = Path(pdf).stem[:30]
                if cache_file.startswith(pdf_stem):
                    pageindex_cached.add(pdf)
                    break

print(f"인덱싱 상태 - Vector: {len(pdf_files)}개 가능, PageIndex 캐시: {len(pageindex_cached)}개")


def compare_answers(selected_docs: list, search_all: bool, question: str, progress=gr.Progress()) -> tuple:
    """Compare answers from both RAG systems with caching."""
    if not question:
        return "", "", ""
    
    # Determine which documents to search
    if search_all:
        docs_to_search = pdf_files
    elif selected_docs:
        docs_to_search = selected_docs
    else:
        return "⚠️ 문서를 선택하거나 '전체 문서 검색'을 체크해주세요.", "", ""
    
    # Check Cache
    history = load_history()
    query_hash = get_query_hash(question, docs_to_search)
    
    if query_hash in history:
        cached = history[query_hash]
        print(f"✅ Cache Hit: {question[:30]}...")
        
        # Format cached responses
        v_res = cached["vector_result"]
        p_res = cached["pageindex_result"]
        
        vector_output = f"**⏱️ {v_res.get('time', 0):.2f}초 (Cached)**\n\n{v_res.get('answer', '')}"
        
        pi_docs = p_res.get("docs_searched", 0)
        pageindex_output = f"**⏱️ {p_res.get('time', 0):.2f}초 (Cached)** ({pi_docs}개 문서 검색)\n\n{p_res.get('answer', '')}"
        
        comparison = f"""---
### 📊 비교 요약
| 항목 | Vector RAG | PageIndex RAG |
|:---|:---:|:---:|
| **응답 시간** | {v_res.get('time', 0):.2f}초 (Cached) | {p_res.get('time', 0):.2f}초 (Cached) |
| **답변 길이** | {len(v_res.get('answer', ''))} 자 | {len(p_res.get('answer', ''))} 자 |
"""
        return vector_output, pageindex_output, comparison, gr.Dropdown(choices=get_recent_queries())
    
    
    results = {}
    
    # Vector RAG
    progress(0.2, desc="Vector RAG 답변 생성 중...")
    try:
        start = time.time()
        vector_answer = vector_rag.answer(question, top_k=5, thinking_effort="medium")
        vector_time = time.time() - start
        results["vector"] = {"answer": vector_answer, "time": vector_time}
    except Exception as e:
        results["vector"] = {"answer": f"❌ 오류: {str(e)}", "time": 0}
    
    # PageIndex RAG
    progress(0.4, desc="PageIndex: 문서 선별 중 (Global Routing)...")
    try:
        start = time.time()
        
        # 1. 문서 선별 (Global Routing)
        available_docs = [d for d in docs_to_search if d in pageindex_cached]
        selected_docs = []
        routing_log = ""
        
        if available_docs:
            try:
                # 라우터로 관련 문서 2개 선별
                selected_docs = pageindex_router.route(question, available_docs, top_k=2)
                routing_log = f"> **🔍 선별된 문서**: " + ", ".join([f"`{os.path.basename(d)[:20]}...`" for d in selected_docs]) + "\n\n"
            except Exception as re:
                print(f"Router error: {re}")
                selected_docs = available_docs # Fallback
        
        all_pageindex_results = []
        # 선별된 문서만 검색
        progress(0.6, desc=f"PageIndex: {len(selected_docs)}개 문서 정밀 검색 중...")
        for doc_name in selected_docs:
            if doc_name in pageindex_cached:
                pdf_path = os.path.join(PDF_DIR, doc_name)
                try:
                    pageindex_rag.build_tree(pdf_path)
                    search_results = pageindex_rag.search(pdf_path, question, top_k=2)
                    for r in search_results:
                        r["source_doc"] = doc_name  # 전체 문서명 사용
                    all_pageindex_results.extend(search_results)
                except:
                    pass
        
        all_pageindex_results = all_pageindex_results[:5]
        
        if all_pageindex_results:
            context_parts = []
            for i, doc in enumerate(all_pageindex_results):
                source = doc.get("source_doc", "Unknown")
                title = doc.get("title", "")
                page = doc.get("page", "?")
                text = doc.get("text", "")[:1500]
                # 문서명을 대괄호로 감싸서 명확히 구분
                context_parts.append(f"[[{source}]] {title} (p.{page})\n{text}")
            
            context = "\n\n".join(context_parts)
            
            system_prompt = """당신은 법률 문서 분석 전문가입니다.
1. 반드시 아래 [검색된 섹션]의 내용만 사용하여 답변하세요.
2. 각 정보의 끝에 반드시 출처를 명시하세요. 형식: `(문서명, p.페이지번호)`
3. 문서명은 파일명 그대로(확장자 포함) 정확하게 기재하세요. 길더라도 생략하지 마세요.
4. 검색된 섹션에서 답을 찾을 수 없으면 "검색된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요."""

            user_prompt = f"""[검색된 섹션]
{context}

[질문]
{question}

[답변]"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            pageindex_answer = pageindex_rag.llm.generate(messages, thinking_effort="medium")
        else:
            pageindex_answer = "검색된 문서에서 관련 정보를 찾을 수 없습니다."
        
        pageindex_time = time.time() - start
        
        final_answer = routing_log + pageindex_answer
        
        results["pageindex"] = {
            "answer": final_answer,
            "time": pageindex_time,
            "docs_searched": len(selected_docs)
        }
    except Exception as e:
        results["pageindex"] = {"answer": f"❌ 오류: {str(e)}", "time": 0, "docs_searched": 0}
    
    progress(1.0, desc="완료!")
    
    # Format responses
    vector_output = f"**⏱️ {results['vector']['time']:.2f}초**\n\n{results['vector']['answer']}"
    
    pi_docs = results["pageindex"].get("docs_searched", 0)
    pageindex_output = f"**⏱️ {results['pageindex']['time']:.2f}초** ({pi_docs}개 문서 검색)\n\n{results['pageindex']['answer']}"
    
    # Comparison summary at the end
    comparison = f"""---
### 📊 비교 요약
| 항목 | Vector RAG | PageIndex RAG |
|:---|:---:|:---:|
| **응답 시간** | {results['vector']['time']:.2f}초 | {results['pageindex']['time']:.2f}초 |
| **답변 길이** | {len(results['vector']['answer'])} 자 | {len(results['pageindex']['answer'])} 자 |
"""
    
    # Save to history
    try:
        history[query_hash] = {
            "timestamp": datetime.now().isoformat(),
            "query": question,
            "selected_docs": list(docs_to_search),
            "vector_result": {
                "answer": results["vector"]["answer"],
                "time": results["vector"]["time"]
            },
            "pageindex_result": {
                "answer": results["pageindex"]["answer"],
                "time": results["pageindex"]["time"],
                "docs_searched": results["pageindex"].get("docs_searched", 0)
            }
        }
        save_history(history)
        print(f"✅ 결과 저장 완료: {query_hash}")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        comparison += f"\n\n🚨 **로깅 실패**: {str(e)}"
    
    return vector_output, pageindex_output, comparison, gr.Dropdown(choices=get_recent_queries())


# Build Gradio UI with sidebar layout
with gr.Blocks(
    title="RAG 비교 테스트 v3",
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
        body, .gradio-container {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .answer-box { 
            border: 1px solid #404040; 
            border-radius: 8px; 
            padding: 16px; 
            background: #2a2a2a;
            min-height: 200px;
            color: #e0e0e0;
        }
        .answer-box p {
            color: #e0e0e0 !important;
        }
        .sidebar { 
            background: #262626; 
            padding: 15px; 
            border-radius: 8px; 
            border: 1px solid #404040;
        }
        /* Markdown headers in Dark Mode */
        h1, h2, h3 { color: #ffffff !important; }
        
        /* Table styles for Dark Mode */
        table { border-color: #404040 !important; }
        th { background-color: #333333 !important; color: #ffffff !important; }
        td { background-color: #2a2a2a !important; color: #e0e0e0 !important; }
    """
) as app:
    
    with gr.Row():
        # ===== 좌측 사이드바: 문서 선택 =====
        with gr.Column(scale=1, elem_classes=["sidebar"]):
            gr.Markdown("## 🕒 최근 질의")
            recent_queries_dropdown = gr.Dropdown(
                choices=get_recent_queries(),
                label="이력 선택",
                interactive=True
            )
            
            gr.Markdown("---")
            gr.Markdown("## 📁 문서 선택")
            
            search_all_checkbox = gr.Checkbox(
                label="🌐 전체 문서 검색",
                value=True,
                info="모든 문서에서 검색"
            )
            
            pdf_multiselect = gr.Dropdown(
                choices=pdf_files,
                multiselect=True,
                label="개별 문서 선택",
                info="전체 검색 해제 시 사용"
            )
            
            gr.Markdown("---")
            gr.Markdown(f"**인덱스 현황**")
            gr.Markdown(f"- Vector: {len(pdf_files)}개")
            gr.Markdown(f"- PageIndex: {len(pageindex_cached)}개")
        
        # ===== 메인 영역: 질문 & 답변 =====
        with gr.Column(scale=3):
            gr.Markdown("# 🔍 Vector RAG vs PageIndex RAG 비교")
            
            # 질문 입력
            question_input = gr.Textbox(
                label="❓ 질문",
                placeholder="예: 인공지능 기본법에서 정의하는 고영향 인공지능은 무엇인가요?",
                lines=2
            )
            
            # Dropdown event handler
            # Dropdown event handler with auto-execution
            def update_input(query):
                return query
            

            
            compare_btn = gr.Button("🚀 비교 분석 실행", variant="primary", size="lg")
            
            gr.Markdown("---")
            
            # 답변 영역
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 Vector RAG")
                    vector_output = gr.Markdown(elem_classes=["answer-box"])
                
                with gr.Column():
                    gr.Markdown("### 🌲 PageIndex RAG")
                    pageindex_output = gr.Markdown(elem_classes=["answer-box"])
            
            # 비교 요약 (맨 마지막)
            comparison_summary = gr.Markdown()
            
            # 예시 질문
            gr.Markdown("---")
            gr.Markdown("### 💡 예시 질문")
            example_questions = gr.Examples(
                examples=[
                    ["인공지능 기본법의 주요 목적은 무엇인가요?"],
                    ["고영향 인공지능이란 무엇인가요?"],
                    ["인공지능 투명성 확보를 위해 어떤 조치가 필요한가요?"],
                    ["인공지능 영향평가는 언제 수행해야 하나요?"],
                ],
                inputs=[question_input]
            )
    
    # Event handlers
    compare_btn.click(
        fn=compare_answers,
        inputs=[pdf_multiselect, search_all_checkbox, question_input],
        outputs=[vector_output, pageindex_output, comparison_summary, recent_queries_dropdown]
    )
    
    # Dropdown event handler (Moved here to avoid NameError)
    recent_queries_dropdown.change(
        fn=update_input, 
        inputs=recent_queries_dropdown, 
        outputs=question_input
    ).then(
        fn=load_cached_result,
        inputs=[recent_queries_dropdown],
        outputs=[vector_output, pageindex_output, comparison_summary, recent_queries_dropdown]
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 RAG 비교 테스트 UI v3 시작")
    print("="*50)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
