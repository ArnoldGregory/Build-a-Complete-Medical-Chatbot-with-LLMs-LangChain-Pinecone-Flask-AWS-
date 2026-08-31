from flask import Flask, render_template, jsonify, request, make_response
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from dotenv import load_dotenv
from src.prompt import *
from src.database import init_db, check_clinic_schedule, book_appointment
from langchain_core.messages import HumanMessage, AIMessage
import os
import uuid


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found in .env file! Check the file exists and has the correct name.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 1. LOCAL CALENDAR STORAGE
init_db()

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

chatModel = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    model="qwen/qwen3.8-27b",
    temperature=0.2,
)

# 2. PATIENT FUNCTION TOOLS
patient_tools = [check_clinic_schedule, book_appointment]

# 3. AGENT INTENT ROUTER
# A single agent holds both capabilities: the two database tools for
# booking/scheduling, and the RAG chain for health/medical questions.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Prompt used by the RAG chain when answering health/medical questions.
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, rag_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def answer_medical_question(question):
    response = rag_chain.invoke({"input": question})
    return response["answer"]


@tool
def answer_health_question(question: str) -> str:
    """Answer a patient's health or medical question using the medical knowledge base. Use this whenever the patient asks about symptoms, conditions, treatments, or any medical question."""
    return answer_medical_question(question)


# Agent tools = booking/scheduling tools + medical Q&A tool
all_tools = patient_tools + [answer_health_question]

agent = create_openai_tools_agent(chatModel, all_tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)

# Per-session chat memory so the conversation flows.
chat_memory = {}


def get_session_id_web():
    sid = request.cookies.get("medibook_session")
    if not sid:
        sid = str(uuid.uuid4())
    return sid


def get_session_id_webhook():
    return request.headers.get("X-Session-Id") or request.args.get("session_id") or str(uuid.uuid4())


def get_history(session_id):
    return chat_memory.setdefault(session_id, [])


def process_message(msg, session_id=None):
    if session_id is None:
        session_id = get_session_id_web()
    history = get_history(session_id)
    result = agent_executor.invoke({"input": msg, "chat_history": list(history)})
    answer = result["output"]
    history.append(HumanMessage(content=msg))
    history.append(AIMessage(content=answer))
    # cap history length to avoid runaway context
    if len(history) > 20:
        del chat_memory[session_id][: len(history) - 20]
    return answer


@app.route("/")
def index():
    resp = make_response(render_template('chat.html'))
    sid = request.cookies.get("medibook_session")
    if not sid:
        resp.set_cookie("medibook_session", str(uuid.uuid4()))
    return resp


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("Patient message : ", msg)
    response = process_message(msg)
    print("Response : ", response)
    resp = make_response(str(response))
    sid = request.cookies.get("medibook_session")
    if sid:
        resp.set_cookie("medibook_session", sid)
    return resp


@app.route("/chat", methods=["POST"])
def chat_webhook():
    data = request.get_json(silent=True) or request.form
    msg = data.get("message", data.get("msg", ""))
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    session_id = get_session_id_webhook()
    response = process_message(msg, session_id)
    return jsonify({"reply": response, "session_id": session_id})


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or request.form
    msg = data.get("message", data.get("msg", ""))
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    session_id = get_session_id_webhook()
    response = process_message(msg, session_id)
    return jsonify({"reply": response, "session_id": session_id})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
