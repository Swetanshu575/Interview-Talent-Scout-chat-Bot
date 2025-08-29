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


murf_client = get_murf_client()

def speak_text_with_murf(text: str, voice_id: str = "en-US-terrell"):
    """Generate TTS audio from text using Murf AI and play in Streamlit."""
    try:
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
MODEL_DEFAULT = "qwen/qwen3-32b"
MODEL_STRICT = "qwen/qwen3-32b"

SYSTEM_INTERVIEWER = """You are TalentScout... (same as before)"""
SYSTEM_EVALUATOR = """You are TalentScout-Eval... (same as before)"""


# -----------------------------
# Core LLM Call
# -----------------------------
def call_llm(client: Groq, model: str, system: str, user: str,
             temperature: float = 0.3, max_tokens: int = 1024) -> str:
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
# (rest of your helper funcs unchanged: extract_json_block, safe_json_parse,
# gen_structured_questions, eval_answer, valid_email, valid_phone, etc.)
# -----------------------------


# -----------------------------
# Streamlit App Views
# -----------------------------
def ask_greeting():
    msg = "Hi! I’m TalentScout. I’ll run a quick initial screen. What’s your **full name**?"
    st.chat_message("assistant").write(msg)
    speak_text_with_murf(msg)


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
                f"Thanks for your time! Overall score: **{overall['average_score']} / 5**  "
                f"(Level: **{overall['level']}**). We’ll review and get back to you."
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
            msg = "Nice to meet you! What’s your **email address**?"
            st.chat_message("assistant").write(msg)
            speak_text_with_murf(msg)
            st.session_state.transcript.append({"role": "assistant", "content": msg})
            st.session_state.assistant_state = "email"

        elif state == "email":
            if valid_email(user_input.strip()):
                c["email"] = user_input.strip()
                msg = "Great—what’s your **phone number**?"
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
            msg = "What’s your **current location** (city, country)?"
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
            with st.spinner("Generating LLM questions…"):
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
            st.chat_message("assistant").write(follow)
            speak_text_with_murf(follow)
            st.session_state.transcript.append({"role": "assistant", "content": follow})

    with st.expander("Settings", expanded=False):
        st.selectbox("Interview Model", [MODEL_DEFAULT, MODEL_STRICT], key="model_interview")
        st.selectbox("Evaluation Model", [MODEL_STRICT, MODEL_DEFAULT], key="model_eval")
    render_progress()

    if st.session_state.assistant_state == "completed":
        export_summary_button()


if __name__ == "__main__":
    main()
