import os
import json
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import streamlit as st
from groq import Groq
from murf import Murf   # <-- Murf SDK for TTS


# =============================
# ⚠️ Setup & Secrets
# =============================

def get_groq_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not api_key:
        st.error("No GROQ_API_KEY found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()
    return Groq(api_key=api_key)


def get_murf_client() -> Murf:
    api_key = st.secrets.get("MURF_API_KEY", os.getenv("MURF_API_KEY", ""))
    if not api_key:
        st.error("No MURF_API_KEY found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()
    return Murf(api_key=api_key)


def speak_text_with_murf(text: str, voice_id: str = "en-US-terrell"):
    """Generate TTS audio from text using Murf AI and play in Streamlit."""
    try:
        murf_client = get_murf_client()
        res = murf_client.text_to_speech.generate(
            text=text,
            voice_id=voice_id,
        )
        audio_url = res.audio_file
        st.audio(audio_url, format="audio/mp3")
    except Exception as e:
        st.warning(f"TTS failed: {e}")


# -----------------------------
# Data Models
# -----------------------------
@dataclass
class TechQuestion:
    id: str
    question: str
    topic: str
    difficulty: str
    rubric: str

@dataclass
class TechAnswer:
    question_id: str
    answer: str
    score: float
    feedback: str


# -----------------------------
# LLM System Prompts
# -----------------------------
MODEL_DEFAULT = "qwen/qwen3-32b"  # Changed to a valid Groq model
MODEL_STRICT = "qwen/qwen3-32b"  # Changed to a valid Groq model

SYSTEM_INTERVIEWER = """You are TalentScout, an expert technical interviewer. Your role is to:
1. Generate relevant technical questions based on candidate's experience and stack
2. Ask clear, specific questions that test practical knowledge
3. Keep questions appropriate to the candidate's experience level
4. Focus on core concepts, problem-solving, and real-world application

Generate questions in JSON format with fields: id, question, topic, difficulty, rubric.
Difficulty levels: "Easy", "Medium", "Hard"
Topics should match the candidate's tech stack.
"""

SYSTEM_EVALUATOR = """You are TalentScout-Eval, an expert technical interview evaluator. Your role is to:
1. Evaluate technical answers objectively and fairly
2. Provide constructive, specific feedback
3. Score answers on a scale of 0-5 based on accuracy, depth, and clarity
4. Be encouraging while maintaining high standards

Provide evaluation in JSON format with fields: score (0-5), feedback (specific comments).
Consider: technical accuracy, completeness, clarity of explanation, practical understanding.
"""


# -----------------------------
# Core LLM Call
# -----------------------------
def call_llm(client: Groq, model: str, system: str, user: str,
             temperature: float = 0.3, max_tokens: int = 1024) -> str:
    try:
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
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return ""


# -----------------------------
# Helper Functions
# -----------------------------
def extract_json_block(text: str) -> str:
    """Extract JSON block from text."""
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*?\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()


def safe_json_parse(text: str) -> Dict[str, Any]:
    """Safely parse JSON from text."""
    try:
        json_text = extract_json_block(text)
        return json.loads(json_text)
    except (json.JSONDecodeError, AttributeError):
        return {}


def gen_structured_questions(client: Groq, model: str, candidate: Dict[str, Any]) -> List[TechQuestion]:
    """Generate structured technical questions."""
    prompt = f"""Generate 5 technical interview questions for this candidate:
    
Name: {candidate.get('name', 'N/A')}
Experience: {candidate.get('experience', 'N/A')} years
Position: {candidate.get('desired_position', 'N/A')}
Tech Stack: {candidate.get('tech_stack', 'N/A')}

Return as JSON array with each question having:
- id: "q1", "q2", etc.
- question: the actual question text
- topic: main technology/concept being tested
- difficulty: "Easy", "Medium", or "Hard"
- rubric: brief scoring criteria

Focus on practical, real-world scenarios relevant to their stack and experience level."""

    try:
        response = call_llm(client, model, SYSTEM_INTERVIEWER, prompt)
        if not response:
            return []
            
        data = safe_json_parse(response)
        questions = []
        
        if isinstance(data, list):
            question_list = data
        else:
            question_list = data.get('questions', [])
            
        for i, q in enumerate(question_list[:5]):
            if isinstance(q, dict) and 'question' in q:
                questions.append(TechQuestion(
                    id=q.get('id', f"q{i+1}"),
                    question=q.get('question', ''),
                    topic=q.get('topic', 'General'),
                    difficulty=q.get('difficulty', 'Medium'),
                    rubric=q.get('rubric', 'Evaluate based on accuracy and clarity')
                ))
        
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []


def eval_answer(client: Groq, model: str, question: TechQuestion, answer: str) -> Dict[str, Any]:
    """Evaluate candidate's answer."""
    prompt = f"""Evaluate this technical interview answer:

Question ({question.topic}, {question.difficulty}): {question.question}
Rubric: {question.rubric}
Candidate Answer: {answer}

Return JSON with:
- score: number from 0-5
- feedback: specific constructive feedback (2-3 sentences)

Consider technical accuracy, completeness, and clarity."""

    try:
        response = call_llm(client, model, SYSTEM_EVALUATOR, prompt)
        if not response:
            return {"score": 0, "feedback": "Could not evaluate answer"}
            
        result = safe_json_parse(response)
        return {
            "score": float(result.get("score", 0)),
            "feedback": result.get("feedback", "No feedback available")
        }
    except Exception as e:
        st.error(f"Error evaluating answer: {e}")
        return {"score": 0, "feedback": "Evaluation failed"}


def valid_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def valid_phone(phone: str) -> bool:
    """Validate phone number (minimum 10 digits)."""
    digits = re.sub(r'\D', '', phone)
    return len(digits) >= 10


def compute_overall() -> Dict[str, Any]:
    """Compute overall interview statistics."""
    answers = st.session_state.get("answers", [])
    if not answers:
        return {
            "average_score": 0.0,
            "level": "No answers",
            "answered": 0,
            "total_questions": len(st.session_state.get("questions", []))
        }
    
    scores = [ans.score for ans in answers]
    avg_score = sum(scores) / len(scores)
    
    if avg_score >= 4.5:
        level = "Excellent"
    elif avg_score >= 3.5:
        level = "Good"
    elif avg_score >= 2.5:
        level = "Fair"
    else:
        level = "Needs Improvement"
    
    return {
        "average_score": round(avg_score, 2),
        "level": level,
        "answered": len(answers),
        "total_questions": len(st.session_state.get("questions", []))
    }


# -----------------------------
# UI Components
# -----------------------------
def header_view():
    """Display app header."""
    st.title("🤖 TalentScout LLM Interview")
    st.markdown("*AI-Powered Technical Screening*")


def sidebar_view():
    """Display sidebar with candidate info and progress."""
    with st.sidebar:
        st.header("📋 Candidate Profile")
        
        candidate = st.session_state.get("candidate", {})
        if candidate.get("name"):
            st.write(f"**Name:** {candidate['name']}")
        if candidate.get("email"):
            st.write(f"**Email:** {candidate['email']}")
        if candidate.get("experience"):
            st.write(f"**Experience:** {candidate['experience']} years")
        if candidate.get("desired_position"):
            st.write(f"**Position:** {candidate['desired_position']}")
        if candidate.get("location"):
            st.write(f"**Location:** {candidate['location']}")
        if candidate.get("tech_stack"):
            st.write(f"**Tech Stack:** {candidate['tech_stack']}")


def render_progress():
    """Display interview progress."""
    if "questions" in st.session_state and st.session_state.questions:
        current = st.session_state.get("current_idx", 0)
        total = len(st.session_state.questions)
        progress = min(current / total, 1.0)
        st.progress(progress, text=f"Question {min(current + 1, total)} of {total}")


def export_summary_button():
    """Provide export functionality."""
    if st.button("📄 Export Interview Summary"):
        candidate = st.session_state.get("candidate", {})
        answers = st.session_state.get("answers", [])
        overall = compute_overall()
        
        summary = {
            "candidate": candidate,
            "answers": [asdict(ans) for ans in answers],
            "overall_score": overall["average_score"],
            "level": overall["level"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.download_button(
            "💾 Download JSON",
            json.dumps(summary, indent=2),
            f"interview_{candidate.get('name', 'candidate')}_{int(time.time())}.json",
            "application/json"
        )


# -----------------------------
# Streamlit App Views
# -----------------------------
def ask_greeting():
    msg = "Hi! I'm TalentScout. I'll run a quick initial screen. What's your **full name**?"
    st.chat_message("assistant").write(msg)
    speak_text_with_murf(msg)


# -----------------------------
# Main App
# -----------------------------
def init_state():
    """Initialize Streamlit session state variables."""
    if "assistant_state" not in st.session_state:
        st.session_state.assistant_state = "greeting"
    if "candidate" not in st.session_state:
        st.session_state.candidate = {}
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "transcript" not in st.session_state:
        st.session_state.transcript = []
    if "model_interview" not in st.session_state:
        st.session_state.model_interview = MODEL_DEFAULT
    if "model_eval" not in st.session_state:
        st.session_state.model_eval = MODEL_STRICT


def main():
    st.set_page_config(page_title="TalentScout LLM Interview", page_icon="🤖", layout="wide")
    init_state()
    client = get_groq_client()

    header_view()
    sidebar_view()

    # Display transcript so far
    for turn in st.session_state.transcript:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    # Initial greeting
    if not st.session_state.transcript:
        ask_greeting()
        st.session_state.transcript.append({
            "role": "assistant", 
            "content": "Hi! I'm TalentScout. I'll run a quick initial screen. What's your **full name**?"
        })

    # Chat input
    user_input = st.chat_input("Type here…")
    if user_input:
        st.session_state.transcript.append({"role": "user", "content": user_input})

        # Graceful exit
        if re.search(r"\b(bye|exit|quit|thanks|thank you)\b", user_input, re.I):
            overall = compute_overall()
            msg = (
                f"Thanks for your time! Overall score: **{overall['average_score']} / 5**  "
                f"(Level: **{overall['level']}**). We'll review and get back to you."
            )
            st.chat_message("assistant").write(msg)
            speak_text_with_murf(msg)
            st.session_state.transcript.append({"role": "assistant", "content": msg})
            export_summary_button()
            st.stop()

        state = st.session_state.assistant_state
        c = st.session_state.candidate

        if state == "greeting":
            c["name"] = user_input.strip()
            msg = "Nice to meet you! What's your **email address**?"
            st.chat_message("assistant").write(msg)
            speak_text_with_murf(msg)
            st.session_state.transcript.append({"role": "assistant", "content": msg})
            st.session_state.assistant_state = "email"

        elif state == "email":
            if valid_email(user_input.strip()):
                c["email"] = user_input.strip()
                msg = "Great—what's your **phone number**?"
                st.chat_message("assistant").write(msg)
                speak_text_with_murf(msg)
                st.session_state.transcript.append({"role": "assistant", "content": msg})
                st.session_state.assistant_state = "phone"
            else:
                msg = "Please provide a valid email (e.g., alex.rao@example.com)"
                st.chat_message("assistant").write(msg)
                speak_text_with_murf(msg)
                st.session_state.transcript.append({"role": "assistant", "content": msg})

        elif state == "phone":
            if valid_phone(user_input.strip()):
                c["phone"] = user_input.strip()
                msg = "How many **years of experience** do you have?"
                st.chat_message("assistant").write(msg)
                speak_text_with_murf(msg)
                st.session_state.transcript.append({"role": "assistant", "content": msg})
                st.session_state.assistant_state = "experience"
            else:
                msg = "Please share a valid phone number (min 10 digits)."
                st.chat_message("assistant").write(msg)
                speak_text_with_murf(msg)
                st.session_state.transcript.append({"role": "assistant", "content": msg})

        elif state == "experience":
            c["experience"] = user_input.strip()
            msg = "Which **role/position** are you applying for?"
            st.chat_message("assistant").write(msg)
            speak_text_with_murf(msg)
            st.session_state.transcript.append({"role": "assistant", "content": msg})
            st.session_state.assistant_state = "position"

        elif state == "position":
            c["desired_position"] = user_input.strip()
            msg = "What's your **current location** (city, country)?"
            st.chat_message("assistant").write(msg)
            speak_text_with_murf(msg)
            st.session_state.transcript.append({"role": "assistant", "content": msg})
            st.session_state.assistant_state = "location"

        elif state == "location":
            c["location"] = user_input.strip()
            msg = ("List your **technical stack** (languages, frameworks, DBs, tools).\n\n"
                   "_Example_: Python, FastAPI, React, PostgreSQL, Docker, AWS")
            st.chat_message("assistant").write(msg)
            speak_text_with_murf(msg)
            st.session_state.transcript.append({"role": "assistant", "content": msg})
            st.session_state.assistant_state = "tech_stack"

        elif state == "tech_stack":
            c["tech_stack"] = user_input.strip()
            with st.spinner("Generating questions…"):
                model_q = st.session_state.model_interview
                qs = gen_structured_questions(client, model_q, c)
            if not qs:
                msg = "I couldn't generate questions. Please re-list your tech stack clearly."
                st.chat_message("assistant").write(msg)
                speak_text_with_murf(msg)
                st.session_state.transcript.append({"role": "assistant", "content": msg})
            else:
                st.session_state.questions = qs
                st.session_state.current_idx = 0
                st.session_state.assistant_state = "questions"
                q = qs[0]
                msg = f"**Q1 ({q.topic}, {q.difficulty})**: {q.question}"
                st.chat_message("assistant").write(msg)
                speak_text_with_murf(msg)
                st.session_state.transcript.append({"role": "assistant", "content": msg})

        elif state == "questions":
            idx = st.session_state.current_idx
            qs = st.session_state.questions
            if idx < len(qs):
                q = qs[idx]
                with st.spinner("Evaluating your answer…"):
                    model_e = st.session_state.model_eval or st.session_state.model_interview
                    result = eval_answer(client, model_e, q, user_input)
                
                ans = TechAnswer(
                    question_id=q.id,
                    answer=user_input,
                    score=float(result.get("score", 0.0)),
                    feedback=result.get("feedback", "")
                )
                st.session_state.answers.append(ans)

                feedback_msg = f"**Score:** {ans.score} / 5\n\n**Feedback:** {ans.feedback}"
                st.chat_message("assistant").write(feedback_msg)
                speak_text_with_murf(feedback_msg)
                st.session_state.transcript.append({"role": "assistant", "content": feedback_msg})

                st.session_state.current_idx += 1
                if st.session_state.current_idx < len(qs):
                    nq = qs[st.session_state.current_idx]
                    msg = f"**Q{st.session_state.current_idx+1} ({nq.topic}, {nq.difficulty})**: {nq.question}"
                    st.chat_message("assistant").write(msg)
                    speak_text_with_murf(msg)
                    st.session_state.transcript.append({"role": "assistant", "content": msg})
                else:
                    st.session_state.assistant_state = "completed"
                    overall = compute_overall()
                    summary = (
                        f"🎉 You're done! Overall score: **{overall['average_score']} / 5**  "
                        f"(Level: **{overall['level']}**).\n\n"
                        f"Answered: {overall['answered']} / {overall['total_questions']} questions.\n\n"
                        "Our team will review your screen and follow up with next steps."
                    )
                    st.chat_message("assistant").write(summary)
                    speak_text_with_murf(summary)
                    st.session_state.transcript.append({"role": "assistant", "content": summary})

        elif state == "completed":
            follow = call_llm(
                client,
                st.session_state.model_interview,
                SYSTEM_INTERVIEWER,
                f"Candidate asked: {user_input}. Provide a concise, professional reply.",
                temperature=0.3,
                max_tokens=250,
            )
            if follow:
                st.chat_message("assistant").write(follow)
                speak_text_with_murf(follow)
                st.session_state.transcript.append({"role": "assistant", "content": follow})

    # Settings and progress
    with st.expander("⚙️ Settings", expanded=False):
        st.selectbox("Interview Model", [MODEL_DEFAULT, MODEL_STRICT], key="model_interview")
        st.selectbox("Evaluation Model", [MODEL_STRICT, MODEL_DEFAULT], key="model_eval")
    
    render_progress()

    if st.session_state.assistant_state == "completed":
        export_summary_button()


if __name__ == "__main__":
    main()
