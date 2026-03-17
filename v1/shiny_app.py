# v1/shiny_app.py
import os
import sys
import shutil
import asyncio
import json
import traceback
from pathlib import Path

from shiny import App, ui, render, reactive
from shiny.types import FileInfo

# -------------------------------------------------------------------------
# Ensure imports work when running as: shiny run v1/shiny_app.py
# -------------------------------------------------------------------------
V1_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V1_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------
# Paths
# ----------------------------
DATA_RAW = V1_ROOT / "data" / "raw"
UPLOADS_DIR = V1_ROOT / "data" / "_uploads"
QUESTIONS_JSON = V1_ROOT / "data" / "questions.json"
QUESTIONS_JSONL = V1_ROOT / "data" / "questions.jsonl"
PROCESSED_DIR = V1_ROOT / "data" / "processed"

for p in [DATA_RAW, UPLOADS_DIR, PROCESSED_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Pipeline imports
# ----------------------------
from v1.app.pipeline import e2e_run

# ----------------------------
# Helpers
# ----------------------------
def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _normalize_q(s: str) -> str:
    s = (s or "").strip().lower()
    for ch in ["\u00a0", "\t", "\r", "\n"]:
        s = s.replace(ch, " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def load_questions() -> list[dict]:
    for path in [QUESTIONS_JSON, QUESTIONS_JSONL]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        if text.startswith("["):
            try:
                data = json.loads(text)
                if not isinstance(data, list):
                    continue
                out = []
                for item in data:
                    q = (item.get("question") or "").strip()
                    if not q:
                        continue
                    out.append({"question": q, "expected_type": item.get("expected_type", "text")})
                return out
            except Exception:
                continue

        try:
            out = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    obj = json.loads(s)
                    q = (obj.get("question") or "").strip()
                    if not q:
                        continue
                    out.append({"question": q, "expected_type": obj.get("expected_type", "text")})
            if out:
                return out
        except Exception:
            continue

    return []


def save_uploaded_file(fileinfo: FileInfo, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fileinfo["datapath"], dest_path)
    return dest_path


def get_latest_processed_contract_dir() -> Path | None:
    if not PROCESSED_DIR.exists():
        return None

    candidates = [p for p in PROCESSED_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def get_results_paths_for_contract_dir(contract_dir: Path | None) -> dict[str, Path | None]:
    if contract_dir is None:
        return {
            "results_jsonl": None,
            "results_json": None,
            "evaluation_summary": None,
            "evaluation_markdown": None,
            "evaluation_html": None,
            "results_pdf": None,
        }

    return {
        "results_jsonl": contract_dir / "results.jsonl",
        "results_json": contract_dir / "results.json",
        "evaluation_summary": contract_dir / "evaluation_summary.json",
        "evaluation_markdown": contract_dir / "evaluation_report.md",
        "evaluation_html": contract_dir / "evaluation_report.html",
        "results_pdf": contract_dir / f"{contract_dir.name}.pdf",
    }


def _build_chunk_index_from_processed(contract_dir: Path | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}

    dirs: list[Path] = []
    if contract_dir and contract_dir.exists():
        dirs = [contract_dir]
    elif PROCESSED_DIR.exists():
        dirs = [p for p in PROCESSED_DIR.iterdir() if p.is_dir()]

    for sub in dirs:
        chunks_file = sub / "chunks.jsonl"
        if not chunks_file.exists():
            continue
        try:
            with chunks_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    chunk_id = obj.get("chunk_id") or obj.get("id") or ""
                    if not chunk_id:
                        continue
                    content = obj.get("content") or obj.get("text") or obj.get("excerpt") or ""
                    page = obj.get("page")
                    out[chunk_id] = {"content": content, "page": page, "raw": obj}
        except Exception:
            continue
    return out


def load_results_cache(contract_dir: Path | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    chunk_index = _build_chunk_index_from_processed(contract_dir)

    resolved_dir = contract_dir or get_latest_processed_contract_dir()
    paths = get_results_paths_for_contract_dir(resolved_dir)

    for path in [paths["results_jsonl"], paths["results_json"]]:
        if path is None or not path.exists():
            continue
        try:
            txt = path.read_text(encoding="utf-8").strip()
            if not txt:
                continue

            if txt.startswith("["):
                arr = json.loads(txt)
                if isinstance(arr, list):
                    for obj in arr:
                        q = (obj.get("question") or "").strip()
                        if not q:
                            continue
                        qn = _normalize_q(q)
                        raw_ans = obj.get("raw_answer") or obj.get("raw") or obj.get("answer") or obj.get("response") or ""
                        val = obj.get("value") if "value" in obj else None
                        citations = obj.get("citations") or obj.get("evidence") or obj.get("top_chunks") or []
                        if isinstance(citations, str):
                            citations = [citations]

                        evidence = []
                        for cid in (citations or []):
                            cstr = str(cid)
                            resolved = chunk_index.get(cstr)
                            evidence.append(
                                {
                                    "chunk_id": cstr,
                                    "page": resolved.get("page") if resolved else None,
                                    "content": resolved.get("content") if resolved else "",
                                }
                            )

                        out[qn] = {
                            "question": q,
                            "expected_type": obj.get("expected_type"),
                            "value": val,
                            "raw_answer": raw_ans,
                            "citations": citations,
                            "evidence": evidence,
                            "raw": obj,
                        }
                continue

            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    q = (obj.get("question") or "").strip()
                    if not q:
                        continue
                    qn = _normalize_q(q)
                    raw_ans = obj.get("raw_answer") or obj.get("raw") or obj.get("answer") or obj.get("response") or ""
                    val = obj.get("value") if "value" in obj else None
                    citations = obj.get("citations") or obj.get("evidence") or obj.get("top_chunks") or []
                    if isinstance(citations, str):
                        citations = [citations]

                    evidence = []
                    for cid in (citations or []):
                        cstr = str(cid)
                        resolved = chunk_index.get(cstr)
                        evidence.append(
                            {
                                "chunk_id": cstr,
                                "page": resolved.get("page") if resolved else None,
                                "content": resolved.get("content") if resolved else "",
                            }
                        )

                    out[qn] = {
                        "question": q,
                        "expected_type": obj.get("expected_type"),
                        "value": val,
                        "raw_answer": raw_ans,
                        "citations": citations,
                        "evidence": evidence,
                        "raw": obj,
                    }
        except Exception:
            continue

    return out


def rag_followup_from_stored_evidence(question: str, chat_history: list[dict], stored_evidence: list[dict]) -> dict:
    aoai_ep = _env("AZURE_OPENAI_ENDPOINT")
    aoai_key = _env("AZURE_OPENAI_API_KEY")
    aoai_ver = _env("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    aoai_chat_deploy = _env("AZURE_DEPLOYMENT_NAME") or _env("AZURE_OPENAI_CHAT_DEPLOYMENT")

    if not (aoai_ep and aoai_key and aoai_chat_deploy):
        if stored_evidence:
            summar = "\n\n".join([(e.get("content") or "")[:800] for e in stored_evidence[:6]])
            return {"answer": summar, "evidence": stored_evidence}
        return {"answer": "No stored evidence and chat backend env vars not set.", "evidence": []}

    from openai import AzureOpenAI
    chat_client = AzureOpenAI(api_key=aoai_key, api_version=aoai_ver, azure_endpoint=aoai_ep)

    ctx_parts = []
    for e in (stored_evidence or [])[:10]:
        cid = e.get("chunk_id") or ""
        content = (e.get("content") or "").strip()
        if content:
            ctx_parts.append(f"[{cid}] {content}")
    context = "\n\n".join(ctx_parts)

    system = (
        "You are a contract intelligence assistant. "
        "Answer using ONLY the provided contract excerpts. "
        "If the answer is not explicitly in the excerpts, say you cannot find it in the provided text. "
        "Cite chunk IDs like [c00012]."
    )

    messages = [{"role": "system", "content": system}]
    for m in (chat_history or [])[-10:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": f"Contract excerpts:\n{context}\n\nQuestion:\n{question}"})

    resp = chat_client.chat.completions.create(
        model=aoai_chat_deploy,
        messages=messages,
        temperature=0.0,
    )
    answer = (resp.choices[0].message.content or "").strip()
    return {"answer": answer, "evidence": stored_evidence}


# ----------------------------
# CSS + Splitter JS
# ----------------------------
BASE_CSS = ui.tags.style("""
html, body { height: 100%; overflow: hidden; }

.ci-compact, .ci-compact * { font-size: 0.92rem; }
.ci-compact h4 { font-size: 1.05rem; margin: 0 0 6px 0; }
.ci-compact h5 { font-size: 1.0rem; margin: 0 0 6px 0; }

.ci-compact .btn { padding: 4px 10px; font-size: 0.90rem; line-height: 1.15; }
.ci-compact .btn-sm { padding: 3px 8px; font-size: 0.86rem; }

.ci-compact .input-group,
.ci-compact .form-control,
.ci-compact .input-group-text,
.ci-compact .btn,
.ci-compact input[type="file"] {
  height: 32px !important;
  min-height: 32px !important;
  line-height: 32px !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}

.ci-progress-outer { width: 100%; height: 9px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-top: 6px; }
.ci-progress-indeterminate { height: 9px; width: 35%; background: #4f46e5; border-radius: 999px; animation: ci-slide 1.1s infinite ease-in-out; }
@keyframes ci-slide { 0% { transform: translateX(-120%); } 100% { transform: translateX(320%); } }
.ci-progress-done { height: 9px; width: 100%; background: #4f46e5; border-radius: 999px; }

.ci-muted { color: #6b7280; font-size: 0.90rem; }

.ci-split { display:flex; width:100%; height: calc(100vh - 120px); border:1px solid #e5e7eb; border-radius:12px; overflow:hidden; }
.ci-pane-left { width:430px; min-width:330px; max-width:70%; display:flex; flex-direction:column; background:#fff; }
.ci-splitter-x { width:10px; cursor:col-resize; background:#f3f4f6; border-left:1px solid #e5e7eb; border-right:1px solid #e5e7eb; }
.ci-pane-right { flex:1; min-width:320px; display:flex; flex-direction:column; background:#fff; }

.ci-left-top { padding:0; height:55%; min-height:240px; max-height:80%; display:flex; flex-direction:column; border-bottom:1px solid #e5e7eb; }
.ci-run-controls { padding:10px; background:#fff; border-bottom:1px solid #f3f4f6; }
.ci-run-body { padding:8px 10px; overflow-y:auto; flex:1 1 auto; }
.ci-run-footer { padding:6px 10px 10px 10px; background:#fff; border-top:1px solid #f3f4f6; }

.ci-splitter-y { height:10px; cursor:row-resize; background:#f3f4f6; border-top:1px solid #e5e7eb; border-bottom:1px solid #e5e7eb; }
.ci-left-bottom { padding:10px; flex:1; min-height:180px; overflow:hidden; display:flex; flex-direction:column; gap:8px; }

.ci-input-col { display:flex; flex-direction:column; height:100%; gap:6px; }
.ci-input-col textarea { flex:1; resize:none; min-height:56px; }
.ci-send-row { display:flex; justify-content:flex-start; gap:6px; }

.ci-right-header { padding:12px; border-bottom:1px solid #e5e7eb; }
.ci-right-body { padding:12px; flex:1; overflow:hidden; }
.ci-chat-wrap { height:100%; overflow-y:auto; padding-right:6px; }
.ci-bubble-user { padding:8px; border:1px solid #e5e7eb; border-radius:10px; margin-bottom:8px; background:#f9fafb; }
.ci-bubble-assistant { padding:8px; border:1px solid #dbeafe; border-radius:10px; margin-bottom:8px; background:#eff6ff; }
""")

SPLITTERS_JS = ui.tags.script("""
(function(){
  function setupSplitters(){
    const split = document.querySelector(".ci-split");
    const left = document.querySelector(".ci-pane-left");
    const splitterX = document.querySelector(".ci-splitter-x");
    if (split && left && splitterX) {
      let draggingX = false;
      splitterX.addEventListener("mousedown", (e) => {
        draggingX = true; document.body.style.userSelect="none"; document.body.style.cursor="col-resize"; e.preventDefault();
      });
      window.addEventListener("mousemove", (e) => {
        if(!draggingX) return;
        const rect = split.getBoundingClientRect();
        const newW = e.clientX - rect.left;
        const minW = 330;
        const maxW = rect.width * 0.70;
        left.style.width = Math.max(minW, Math.min(maxW, newW)) + "px";
      });
      window.addEventListener("mouseup", () => {
        if(!draggingX) return;
        draggingX=false; document.body.style.userSelect=""; document.body.style.cursor="";
      });
    }

    const leftPane = document.querySelector(".ci-pane-left");
    const topPane = document.querySelector(".ci-left-top");
    const splitterY = document.querySelector(".ci-splitter-y");
    if (leftPane && topPane && splitterY) {
      let draggingY = false;
      splitterY.addEventListener("mousedown", (e) => {
        draggingY=true; document.body.style.userSelect="none"; document.body.style.cursor="row-resize"; e.preventDefault();
      });
      window.addEventListener("mousemove", (e) => {
        if(!draggingY) return;
        const rect = leftPane.getBoundingClientRect();
        const newH = e.clientY - rect.top;
        const minH = 220;
        const maxH = rect.height - 170;
        topPane.style.height = Math.max(minH, Math.min(maxH, newH)) + "px";
      });
      window.addEventListener("mouseup", () => {
        if(!draggingY) return;
        draggingY=false; document.body.style.userSelect=""; document.body.style.cursor="";
      });
    }
  }
  window.addEventListener("load", setupSplitters);
  setTimeout(setupSplitters, 800);
})();
""")

# ----------------------------
# UI builders
# ----------------------------
def left_pane_ui():
    return ui.tags.div(
        ui.tags.div(
            ui.tags.div(
                ui.tags.h4("Upload contract"),
                ui.input_file("pdf", "Choose PDF", accept=[".pdf"], multiple=False),
                ui.tags.div(style="height:6px;"),
                ui.output_ui("run_button_ui"),
                ui.output_ui("progress_ui"),
                ui.tags.div(style="height:6px;"),
                ui.output_text_verbatim("status"),
                class_="ci-run-controls ci-compact",
            ),
            ui.tags.div(ui.div(), class_="ci-run-body ci-compact"),
            ui.tags.div(ui.output_ui("download_block"), class_="ci-run-footer ci-compact"),
            class_="ci-left-top",
        ),
        ui.tags.div(class_="ci-splitter-y"),
        ui.tags.div(
            ui.tags.div(ui.tags.h4("Chat"), class_="ci-compact"),
            ui.tags.div(
                ui.input_text_area("chat_input", "", rows=2, placeholder="ask a question"),
                ui.tags.div(
                    ui.input_action_button("send", "Send", class_="btn-primary btn-sm"),
                    class_="ci-send-row",
                ),
                ui.output_text("chat_error"),
                class_="ci-input-col ci-compact",
            ),
            class_="ci-left-bottom",
        ),
        class_="ci-pane-left",
    )


def right_pane_ui():
    return ui.tags.div(
        ui.tags.div(ui.tags.h4("Answers"), class_="ci-right-header"),
        ui.tags.div(ui.output_ui("chat_messages"), class_="ci-right-body"),
        class_="ci-pane-right",
    )


def home_tab():
    return ui.page_fluid(
        BASE_CSS,
        SPLITTERS_JS,
        ui.tags.div(left_pane_ui(), ui.tags.div(class_="ci-splitter-x"), right_pane_ui(), class_="ci-split"),
    )


def questions_tab():
    return ui.page_fluid(
        BASE_CSS,
        ui.card(
            ui.h4("Questions"),
            ui.output_ui("questions_summary"),
            ui.br(),
            ui.input_switch("show_questions", "View questions", value=False),
            ui.br(),
            ui.output_ui("questions_list"),
        ),
    )


app_ui = ui.page_navbar(
    ui.nav_panel("Home", home_tab()),
    ui.nav_panel("Questions", questions_tab()),
    title="IMC Contract Intelligence Assistant",
)

# ----------------------------
# Background pipeline task
# ----------------------------
@reactive.extended_task
async def pipeline_task(saved_pdf_path: str) -> dict:
    def _run_all(saved_pdf_path_inner: str):
        try:
            print(f"[pipeline_task] started for uploaded file: {saved_pdf_path_inner}")

            e2e_run.main()
            print("[pipeline_task] full e2e pipeline completed")

            raw_files = [f for f in DATA_RAW.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
            if not raw_files:
                return {"error": True, "message": "No PDF found in raw after pipeline run."}

            raw_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            contract_id = raw_files[0].stem
            processed_dir = PROCESSED_DIR / contract_id
            paths = get_results_paths_for_contract_dir(processed_dir)

            return {
                "contract_id": contract_id,
                "processed_dir": str(processed_dir),
                "results_jsonl": str(paths["results_jsonl"]) if paths["results_jsonl"] else None,
                "results_json": str(paths["results_json"]) if paths["results_json"] else None,
                "evaluation_summary": str(paths["evaluation_summary"]) if paths["evaluation_summary"] else None,
                "evaluation_markdown": str(paths["evaluation_markdown"]) if paths["evaluation_markdown"] else None,
                "evaluation_html": str(paths["evaluation_html"]) if paths["evaluation_html"] else None,
                "results_pdf": str(paths["results_pdf"]) if paths["results_pdf"] else None,
            }
        except Exception as e:
            tb = traceback.format_exc()
            print("[pipeline_task] ERROR:", str(e))
            print(tb)
            return {"error": True, "message": str(e), "traceback": tb}

    return await asyncio.to_thread(_run_all, saved_pdf_path)

# ----------------------------
# Server
# ----------------------------
def server(input, output, session):
    state = reactive.Value("idle")
    message = reactive.Value("")
    started = reactive.Value(False)

    results_cache = reactive.Value({})
    latest_contract_dir = reactive.Value(None)
    latest_run_info = reactive.Value({})

    chat_err = reactive.Value("")
    anchor_q = reactive.Value(None)
    chat_msgs = reactive.Value([])

    def set_state(st: str, msg: str = ""):
        state.set(st)
        message.set(msg)

    initial_contract_dir = get_latest_processed_contract_dir()
    latest_contract_dir.set(initial_contract_dir)
    results_cache.set(load_results_cache(initial_contract_dir))

    @reactive.effect
    @reactive.event(input.run)
    def _on_run():
        if state.get() == "running":
            return

        pdf_files = input.pdf()
        if not pdf_files:
            started.set(False)
            set_state("error", "Please upload a PDF first.")
            return

        pdf_info = pdf_files[0]

        for f in DATA_RAW.glob("*.pdf"):
            try:
                f.unlink()
            except Exception:
                pass

        dest_pdf = DATA_RAW / pdf_info["name"]
        save_uploaded_file(pdf_info, dest_pdf)

        started.set(True)
        set_state("running", "")
        latest_run_info.set({})
        pipeline_task.invoke(str(dest_pdf))

    @reactive.effect
    def _watch_task():
        st = pipeline_task.status()
        if st == "running":
            return
        if st == "error":
            set_state("error", "Pipeline failed.")
            return
        if st == "success":
            res = pipeline_task.result()
            if isinstance(res, dict) and res.get("error"):
                set_state("error", f"Pipeline error: {res.get('message')}")
                return

            latest_run_info.set(res or {})
            contract_dir_str = (res or {}).get("processed_dir")
            contract_dir = Path(contract_dir_str) if contract_dir_str else get_latest_processed_contract_dir()
            latest_contract_dir.set(contract_dir)
            results_cache.set(load_results_cache(contract_dir))
            set_state("done", "")
            return

    @output
    @render.ui
    def run_button_ui():
        if state.get() == "running":
            return ui.tags.button(
                "Run",
                class_="btn btn-primary btn-sm",
                disabled=True,
                style="opacity:0.6; cursor:not-allowed; width:110px;",
            )
        return ui.input_action_button("run", "Run", class_="btn-primary btn-sm")

    @output
    @render.ui
    def progress_ui():
        if not started.get():
            return ui.div()
        st = state.get()
        if st == "running":
            return ui.div(
                ui.tags.div("Processing...", style="margin-top:4px;"),
                ui.tags.div("⏳ Running", class_="ci-muted", style="margin-top:2px;"),
                ui.tags.div(ui.tags.div(class_="ci-progress-indeterminate"), class_="ci-progress-outer"),
            )
        if st == "done":
            return ui.div(
                ui.tags.div(ui.tags.div(class_="ci-progress-done"), class_="ci-progress-outer"),
            )
        if st == "error":
            return ui.div(
                ui.tags.div(ui.tags.div(class_="ci-progress-done"), class_="ci-progress-outer"),
            )
        return ui.div()

    @output
    @render.text
    def status():
        st = state.get()
        msg = message.get()
        if st == "error":
            return msg
        return ""

    @output
    @render.ui
    def download_block():
        contract_dir = latest_contract_dir.get()
        paths = get_results_paths_for_contract_dir(contract_dir)

        if (
            state.get() == "done"
            and paths["results_pdf"]
            and paths["results_pdf"].exists()
        ):
            return ui.download_button(
                "download_results_pdf",
                "Download Results PDF",
                class_="btn-primary btn-sm",
            )

        return ui.div()

    @output
    @render.download(filename=lambda: f"{latest_contract_dir.get().name}.pdf" if latest_contract_dir.get() else "results.pdf")
    def download_results_pdf():
        def _iter():
            contract_dir = latest_contract_dir.get()
            paths = get_results_paths_for_contract_dir(contract_dir)

            if paths["results_pdf"] and paths["results_pdf"].exists():
                with open(paths["results_pdf"], "rb") as f:
                    yield f.read()
            else:
                yield b""

        return _iter()

    @output
    @render.ui
    def questions_summary():
        qs = load_questions()
        return ui.p(f"There are {len(qs)} questions to extract.") if qs else ui.p("There are 0 questions to extract.")

    @output
    @render.ui
    def questions_list():
        if not input.show_questions():
            return ui.div()
        qs = load_questions()
        if not qs:
            return ui.p("No questions loaded.")

        rows = [ui.tags.li(ui.tags.span(q["question"])) for q in qs]

        return ui.tags.div(
            ui.tags.ol(*rows),
            style="height: calc(100vh - 260px); overflow-y: auto; border: 1px solid #e5e7eb; padding: 10px; border-radius: 8px;",
        )

    @reactive.effect
    @reactive.event(input.send)
    def _send_chat():
        q = (input.chat_input() or "").strip()
        if not q:
            return

        chat_err.set("")
        msgs = list(chat_msgs.get())
        msgs.append({"role": "user", "content": q})

        cache = results_cache.get()
        key = _normalize_q(q)

        if key in cache:
            row = cache[key]
            anchor_q.set(row["question"])
            msgs.append({"role": "assistant", "content": str(row.get("raw_answer") or row.get("value") or "")})
            chat_msgs.set(msgs)
            ui.update_text_area("chat_input", value="")
            return

        found = None
        for k, v in cache.items():
            if k.startswith(key) or key.startswith(k):
                found = v
                break

        if found:
            anchor_q.set(found["question"])
            msgs.append({"role": "assistant", "content": str(found.get("raw_answer") or found.get("value") or "")})
            chat_msgs.set(msgs)
            ui.update_text_area("chat_input", value="")
            return

        aq = anchor_q.get()
        if aq and _normalize_q(aq) in cache:
            stored_ev = cache[_normalize_q(aq)].get("evidence", [])
            history = [{"role": m["role"], "content": m["content"]} for m in msgs if m.get("role") in ("user", "assistant")]
            out = rag_followup_from_stored_evidence(q, history, stored_ev)
            msgs.append({"role": "assistant", "content": (out.get("answer") or "").strip()})
            chat_msgs.set(msgs)
            ui.update_text_area("chat_input", value="")
            return

        msgs.append({
            "role": "assistant",
            "content": "I couldn't find a matching precomputed answer. Ask one of the predefined questions first (Questions tab) to anchor the chat, then ask follow-ups.",
        })
        chat_msgs.set(msgs)
        ui.update_text_area("chat_input", value="")

    @output
    @render.text
    def chat_error():
        return chat_err.get()

    @output
    @render.ui
    def chat_messages():
        msgs = list(chat_msgs.get())
        if not msgs:
            return ui.p("ask a question", class_="ci-muted")

        blocks = []
        for m in msgs:
            if m.get("role") == "user":
                blocks.append(
                    ui.tags.div(
                        ui.tags.div("You", style="font-weight:600;"),
                        ui.tags.div(m.get("content", "")),
                        class_="ci-bubble-user",
                    )
                )
            else:
                blocks.append(
                    ui.tags.div(
                        ui.tags.div("Assistant", style="font-weight:600;"),
                        ui.tags.div(m.get("content", "")),
                        class_="ci-bubble-assistant",
                    )
                )

        return ui.tags.div(*blocks, class_="ci-chat-wrap")


app = App(app_ui, server)