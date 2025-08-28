import os
import json
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import streamlit as st
from groq import Groq

# =============================
# ⚠️ Setup & Secrets
# -----------------------------
# Do NOT hardcode your API key. Set it in one of these ways before running:
#   1) Streamlit Secrets:  st.secrets["GROQ_API_KEY"]
#   2) Environment Var:    export GROQ_API_KEY=your_key
# The app will try both.
# =============================

def get_groq_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not api_key:
        st.error("No GROQ_API_KEY found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()
    return Groq(api_key=api_key)

# -----------------------------
# Data Models
# -----------------------------
@dataclass
class TechQuestion:
    id: str
    question: str
    topic: str
    difficulty: str  # easy | medium | hard
    rubric: str      # short rubric/checklist for grading

@dataclass
class TechAnswer:
    question_id: str
    answer: str
    score: float
    feedback: str

# -----------------------------
# LLM Utilities
# -----------------------------
MODEL_DEFAULT = "qwen/qwen3-32b"
MODEL_STRICT = "qwen/qwen3-32b"  # used for evaluation if available; fall back if not

SYSTEM_INTERVIEWER = (
    """
You are TalentScout, a senior technical interviewer and hiring assistant for a technology recruitment agency.
Your goals:
1) Conduct an initial screen that feels warm, concise, and professional.
2) Collect essential candidate details (name, email, phone, experience, location, desired role, tech stack).
3) Generate 4-5 highly relevant, practical technical questions based on the candidate's tech stack & experience.
4) Each question MUST include: topic, difficulty (easy|medium|hard), and a concise grading rubric.
5) Keep language clear, direct, and free of fluff.
6) Return machine-readable JSON only when asked to return JSON.
    """.strip()
)

SYSTEM_EVALUATOR = (
    """
You are TalentScout-Eval, a rigorous but fair senior engineer.
Evaluate candidate answers against the provided rubric.
Return a JSON object with fields: {"score": 0-5 (float), "feedback": "string"}.
Score guidance:
- 0-1: Incorrect, lacks understanding
- 2: Partially correct but major gaps
- 3: Mostly correct with minor gaps
- 4: Correct and well-explained
- 5: Excellent, precise, production-aware
Keep feedback concise (<= 120 words) and actionable.
    """.strip()
)

JSON_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "question": {"type": "string"},
                    "topic": {"type": "string"},
                    "difficulty": {"type": "string"},
                    "rubric": {"type": "string"}
                },
                "required": ["id", "question", "topic", "difficulty", "rubric"]
            },
            "minItems": 3,
            "maxItems": 6
        }
    },
    "required": ["questions"]
}

# -----------------------------
# Core LLM Call
# -----------------------------

def call_llm(client: Groq, model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content

# -----------------------------
# JSON Helpers
# -----------------------------

def extract_json_block(text: str) -> Optional[str]:
    """Extract the first JSON block from a string."""
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else None


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def gen_structured_questions(client: Groq, model: str, candidate: Dict[str, Any]) -> List[TechQuestion]:
    user = (
        f"Return ONLY JSON per schema. Candidate profile: "
        f"name={candidate.get('name')}, email={candidate.get('email')}, phone={candidate.get('phone')},\n"
        f"experience-years={candidate.get('experience')}, desired-role={candidate.get('desired_position')},\n"
        f"location={candidate.get('location')}, tech-stack={candidate.get('tech_stack')}\n\n"
        f"Schema: {json.dumps(JSON_QUESTIONS_SCHEMA)}"
    )
    raw = call_llm(client, model, SYSTEM_INTERVIEWER, user, temperature=0.2, max_tokens=1200)
    blob = extract_json_block(raw) or raw
    data = safe_json_parse(blob)
    if not data or "questions" not in data:
        # simple fallback with a second attempt
        raw2 = call_llm(client, model, SYSTEM_INTERVIEWER, user + "\n\nReturn STRICT JSON only.", temperature=0.1, max_tokens=1200)
        blob2 = extract_json_block(raw2) or raw2
        data = safe_json_parse(blob2)
    questions: List[TechQuestion] = []
    if data and "questions" in data:
        for q in data["questions"]:
            try:
                questions.append(
                    TechQuestion(
                        id=str(q.get("id") or f"q-{len(questions)+1}"),
                        question=q.get("question", "").strip(),
                        topic=q.get("topic", "General").strip(),
                        difficulty=q.get("difficulty", "medium").strip().lower(),
                        rubric=q.get("rubric", "Relevant, correct, concise").strip(),
                    )
                )
            except Exception:
                continue
    return questions[:5] if questions else []


def eval_answer(client: Groq, model: str, question: TechQuestion, answer: str) -> Dict[str, Any]:
    user = (
        f"Question: {question.question}\n"
        f"Topic: {question.topic}\n"
        f"Difficulty: {question.difficulty}\n"
        f"Rubric: {question.rubric}\n"
        f"Candidate Answer: {answer}\n\n"
        f"Return JSON with fields: score (0-5 float) and feedback (string)."
    )
    raw = call_llm(client, model, SYSTEM_EVALUATOR, user, temperature=0.0, max_tokens=400)
    blob = extract_json_block(raw) or raw
    data = safe_json_parse(blob)
    if not data or "score" not in data:
        raw2 = call_llm(client, model, SYSTEM_EVALUATOR, user + "\nReturn STRICT JSON only.", temperature=0.0, max_tokens=400)
        blob2 = extract_json_block(raw2) or raw2
        data = safe_json_parse(blob2)
    if not data:
        data = {"score": 0.0, "feedback": "Evaluation service temporarily unavailable."}
    # clamp score
    try:
        data["score"] = max(0.0, min(5.0, float(data.get("score", 0.0))))
    except Exception:
        data["score"] = 0.0
    return data

# -----------------------------
# Validation Helpers
# -----------------------------
EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_RE = re.compile(r'^[\+]?\d[\d\-\s\(\)]{9,}$')


def valid_email(x: str) -> bool:
    return bool(EMAIL_RE.match(x or ""))


def valid_phone(x: str) -> bool:
    if not x:
        return False
    clean = re.sub(r"[\s\-\(\)]", "", x)
    return bool(PHONE_RE.match(x)) and len(clean) >= 10

# -----------------------------
# Streamlit App
# -----------------------------

def init_state():
    if "assistant_state" not in st.session_state:
        st.session_state.assistant_state = "greeting"
    if "candidate" not in st.session_state:
        st.session_state.candidate = {}
    if "questions" not in st.session_state:
        st.session_state.questions: List[TechQuestion] = []
    if "answers" not in st.session_state:
        st.session_state.answers: List[TechAnswer] = []
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "transcript" not in st.session_state:
        st.session_state.transcript: List[Dict[str, str]] = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ts-{int(time.time())}"
    if "model_interview" not in st.session_state:
        st.session_state.model_interview = MODEL_DEFAULT
    if "model_eval" not in st.session_state:
        st.session_state.model_eval = MODEL_STRICT


def sidebar_view():
    st.sidebar.header("📋 Process Overview")
    st.sidebar.write(
        "1. Personal Info → 2. Tech Stack → 3. LLM Interview → 4. Evaluation & Score → 5. Summary"
    )
    c = st.session_state.candidate
    if c:
        st.sidebar.subheader("📊 Candidate")
        for k, v in c.items():
            if v:
                st.sidebar.write(f"**{k.replace('_',' ').title()}:** {v}")
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: type 'bye' or 'exit' to wrap up.")


def header_view():
    st.markdown("""
    <style>
      .main-header { font-size: 2.2rem; text-align:center; margin-bottom: .25rem; }
      .sub-header { text-align:center; color:#666; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-header">🤖 TalentScout — LLM Interview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered initial screening with on-the-fly question generation & scoring</div>', unsafe_allow_html=True)


def export_summary_button():
    candidate = st.session_state.candidate
    questions = [asdict(q) for q in st.session_state.questions]
    answers = [asdict(a) for a in st.session_state.answers]
    overall = compute_overall()
    summary = {
        "session_id": st.session_state.session_id,
        "candidate": candidate,
        "questions": questions,
        "answers": answers,
        "overall": overall,
        "transcript": st.session_state.transcript,
    }
    st.download_button(
        label="⬇️ Download Summary (JSON)",
        file_name=f"talentscout_summary_{st.session_state.session_id}.json",
        mime="application/json",
        data=json.dumps(summary, indent=2),
        use_container_width=True,
    )

# -----------------------------
# Flow Logic
# -----------------------------

def ask_greeting():
    st.chat_message("assistant").write(
        "Hi! I’m TalentScout. I’ll run a quick initial screen. What’s your **full name**?"
    )


def compute_overall() -> Dict[str, Any]:
    scores = [a.score for a in st.session_state.answers]
    avg = round(sum(scores) / len(scores), 2) if scores else 0.0
    level = (
        "Needs Improvement" if avg < 2.0 else
        "Promising" if avg < 3.0 else
        "Strong" if avg < 4.0 else
        "Excellent"
    )
    return {"average_score": avg, "level": level, "answered": len(scores), "total_questions": len(st.session_state.questions)}


def render_progress():
    state = st.session_state.assistant_state
    steps = ["greeting","email","phone","experience","position","location","tech_stack","questions","completed"]
    idx = max(steps.index(state), 0) if state in steps else 0
    st.progress(idx / (len(steps) - 1))
    if state == "questions" and st.session_state.questions:
        st.caption(
            f"Question {st.session_state.current_idx + 1} of {len(st.session_state.questions)}"
        )


# -----------------------------
# Main App
# -----------------------------

def main():
    st.set_page_config(page_title="TalentScout LLM Interview", page_icon="🤖", layout="wide")
    init_state()
    client = get_groq_client()

    header_view()
    sidebar_view()

    # display transcript so far
    for turn in st.session_state.transcript:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    # initial greeting
    if not st.session_state.transcript:
        ask_greeting()
        st.session_state.transcript.append({"role": "assistant", "content": "Hi! I’m TalentScout. I’ll run a quick initial screen. What’s your **full name**?"})

    # chat input
    user_input = st.chat_input("Type here…")
    if user_input:
        st.session_state.transcript.append({"role": "user", "content": user_input})

        # graceful exit
        if re.search(r"\b(bye|exit|quit|thanks|thank you)\b", user_input, re.I):
            overall = compute_overall()
            msg = (
                f"Thanks for your time! Overall score: **{overall['average_score']} / 5**  (Level: **{overall['level']}**). "
                f"We’ll review and get back to you."
            )
            st.chat_message("assistant").write(msg)
            st.session_state.transcript.append({"role": "assistant", "content": msg})
            export_summary_button()
            st.stop()

        state = st.session_state.assistant_state
        c = st.session_state.candidate

        # FSM
        if state == "greeting":
            c["name"] = user_input.strip()
            st.chat_message("assistant").write("Nice to meet you! What’s your **email address**?")
            st.session_state.transcript.append({"role": "assistant", "content": "Nice to meet you! What’s your **email address**?"})
            st.session_state.assistant_state = "email"

        elif state == "email":
            if valid_email(user_input.strip()):
                c["email"] = user_input.strip()
                st.chat_message("assistant").write("Great—what’s your **phone number**?")
                st.session_state.transcript.append({"role": "assistant", "content": "Great—what’s your **phone number**?"})
                st.session_state.assistant_state = "phone"
            else:
                st.chat_message("assistant").write("Please provide a valid email (e.g., alex.rao@example.com)")
                st.session_state.transcript.append({"role": "assistant", "content": "Please provide a valid email (e.g., alex.rao@example.com)"})

        elif state == "phone":
            if valid_phone(user_input.strip()):
                c["phone"] = user_input.strip()
                st.chat_message("assistant").write("How many **years of experience** do you have?")
                st.session_state.transcript.append({"role": "assistant", "content": "How many **years of experience** do you have?"})
                st.session_state.assistant_state = "experience"
            else:
                st.chat_message("assistant").write("Please share a valid phone number (min 10 digits).")
                st.session_state.transcript.append({"role": "assistant", "content": "Please share a valid phone number (min 10 digits)."})

        elif state == "experience":
            c["experience"] = user_input.strip()
            st.chat_message("assistant").write("Which **role/position** are you applying for?")
            st.session_state.transcript.append({"role": "assistant", "content": "Which **role/position** are you applying for?"})
            st.session_state.assistant_state = "position"

        elif state == "position":
            c["desired_position"] = user_input.strip()
            st.chat_message("assistant").write("What’s your **current location** (city, country)?")
            st.session_state.transcript.append({"role": "assistant", "content": "What’s your **current location** (city, country)?"})
            st.session_state.assistant_state = "location"

        elif state == "location":
            c["location"] = user_input.strip()
            st.chat_message("assistant").write(
                "List your **technical stack** (languages, frameworks, DBs, tools).\n\n"
                "_Example_: Python, FastAPI, React, PostgreSQL, Docker, AWS"
            )
            st.session_state.transcript.append({"role": "assistant", "content": "List your **technical stack** (languages, frameworks, DBs, tools).\n\n_Example_: Python, FastAPI, React, PostgreSQL, Docker, AWS"})
            st.session_state.assistant_state = "tech_stack"

        elif state == "tech_stack":
            c["tech_stack"] = user_input.strip()
            with st.spinner("Generating LLM questions…"):
                model_q = st.session_state.model_interview
                qs = gen_structured_questions(client, model_q, c)
            if not qs:
                st.chat_message("assistant").write("I couldn't generate questions. Please re-list your tech stack clearly.")
                st.session_state.transcript.append({"role": "assistant", "content": "I couldn't generate questions. Please re-list your tech stack clearly."})
            else:
                st.session_state.questions = qs
                st.session_state.current_idx = 0
                st.session_state.assistant_state = "questions"
                q = qs[0]
                st.chat_message("assistant").write(
                    f"**Q1 ({q.topic}, {q.difficulty})**: {q.question}"
                )
                st.session_state.transcript.append({"role": "assistant", "content": f"**Q1 ({q.topic}, {q.difficulty})**: {q.question}"})

        elif state == "questions":
            idx = st.session_state.current_idx
            qs = st.session_state.questions
            if idx < len(qs):
                q = qs[idx]
                # Evaluate the answer
                with st.spinner("Evaluating your answer…"):
                    model_e = st.session_state.model_eval or st.session_state.model_interview
                    result = eval_answer(client, model_e, q, user_input)
                ans = TechAnswer(question_id=q.id, answer=user_input, score=float(result.get("score", 0.0)), feedback=result.get("feedback", ""))
                st.session_state.answers.append(ans)

                # show feedback
                st.chat_message("assistant").write(
                    f"**Score:** {ans.score} / 5\n\n**Feedback:** {ans.feedback}"
                )
                st.session_state.transcript.append({"role": "assistant", "content": f"**Score:** {ans.score} / 5\n\n**Feedback:** {ans.feedback}"})

                # move to next
                st.session_state.current_idx += 1
                if st.session_state.current_idx < len(qs):
                    nq = qs[st.session_state.current_idx]
                    st.chat_message("assistant").write(
                        f"**Q{st.session_state.current_idx+1} ({nq.topic}, {nq.difficulty})**: {nq.question}"
                    )
                    st.session_state.transcript.append({"role": "assistant", "content": f"**Q{st.session_state.current_idx+1} ({nq.topic}, {nq.difficulty})**: {nq.question}"})
                else:
                    # completed
                    st.session_state.assistant_state = "completed"
                    overall = compute_overall()
                    summary = (
                        f"🎉 You're done! Overall score: **{overall['average_score']} / 5**  (Level: **{overall['level']}**).\n\n"
                        f"Answered: {overall['answered']} / {overall['total_questions']} questions.\n\n"
                        "Our team will review your screen and follow up with next steps."
                    )
                    st.chat_message("assistant").write(summary)
                    st.session_state.transcript.append({"role": "assistant", "content": summary})

        elif state == "completed":
            # Optional post-completion Q&A: summarize or clarify using interviewer persona
            follow = call_llm(
                client,
                st.session_state.model_interview,
                SYSTEM_INTERVIEWER,
                f"Candidate asked: {user_input}. Provide a concise, professional reply.",
                temperature=0.3,
                max_tokens=250,
            )
            st.chat_message("assistant").write(follow)
            st.session_state.transcript.append({"role": "assistant", "content": follow})

    # Controls & Progress
    with st.expander("Settings", expanded=False):
        st.selectbox("Interview Model", [MODEL_DEFAULT, MODEL_STRICT], key="model_interview")
        st.selectbox("Evaluation Model", [MODEL_STRICT, MODEL_DEFAULT], key="model_eval")
    render_progress()

    if st.session_state.assistant_state == "completed":
        export_summary_button()


if __name__ == "__main__":
    main()
