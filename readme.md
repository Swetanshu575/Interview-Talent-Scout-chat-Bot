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
   pip install -r requirements.txt
2. Create a folder name .streamlit and then Create a file name secrets.toml inside the .streamlit:
3. Now Access the Groq API key and paste it in Secrets.toml using VS Code editor and then Paste in this format :-  GROQ_API_KEY = "your api key "
4. Running the code :-
   ```bash
   streamlit run TalentScoutmain.py
   Programming Language: Python 3.8+
Libraries:streamlit: Frontend UI and dashboard.
pandas, plotly: Data analytics and visualization.
re: Input validation (email, phone).
json: Simulated data and chat history storage.
logging: Debugging and logging.

LLM:used Groq Api
Architecture:Modular design with hiring_assistant.py for chatbot logic and dashboard.py for the main app.
Uses Streamlit's session state for conversation context and JSON files for persistent storage.

Data Privacy: Stores anonymized data locally, ensuring GDPR compliance.

Prompt DesignInformation Collection: Sequential prompts (e.g., "Please provide your full name.") ensure structured data gathering.
Technical Questions: Dynamic prompts generate questions based on tech stack (e.g., "Generate technical questions for Python, Django").
Context Handling: Conversation history stored in st.session_state.conversation and chat_history.json.
Fallback: Handles unclear inputs with clarification requests.

Challenges & SolutionsChallenge: Persistent chat history across sessions.Solution: Implemented JSON-based storage (chat_history.json) with timestamps.

Challenge: Integrating LLM for dynamic responses.Solution: Simulated LLM with rule-based question generation, extensible to xAI's API.

Challenge: Building an intuitive dashboard.Solution: Used Streamlit multi-page navigation and Plotly for visualizations.

Challenge: Data privacy compliance.Solution: Simulated storage with no real data persistence, validated inputs for quality.


