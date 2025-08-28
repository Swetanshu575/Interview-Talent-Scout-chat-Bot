# TalentScout Hiring Assistant Dashboard

## Project Overview
The TalentScout Hiring Assistant Dashboard is an enhanced version of the Hiring Assistant chatbot, designed for a fictional recruitment agency specializing in technology placements. Built using **Streamlit**, it integrates a chatbot for candidate screening with a dashboard for analytics and chat history. The chatbot uses a simulated large language model (LLM) for dynamic responses and technical question generation, with persistent chat history stored in a JSON file.

## Features
- **Chatbot**:
  - Greets candidates and collects information (name, email, phone, experience, position, location, tech stack).
  - Generates 3-5 technical questions based on the candidate's tech stack.
  - Maintains conversation context and handles unexpected inputs.
  - Ends conversations gracefully with next steps.
- **Chat History**: Persists all interactions in `chat_history.json` with timestamps.

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/talentscout-hiring-assistant.git
