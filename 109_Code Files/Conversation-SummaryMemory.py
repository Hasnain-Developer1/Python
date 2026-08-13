from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationSummaryMemory

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

summary_memory = ConversationSummaryMemory(llm=llm)

def chat(user_text):
    history = summary_memory.load_memory_variables({}).get("history", "")

    prompt = f"""
You are a helpful assistant.

Conversation summary so far:
{history}

User: {user_text}

Assistant:
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        text = ""
        for item in response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "")
    else:
        text = response.content

    summary_memory.save_context(
        {"input": user_text},
        {"output": text}
    )

    return text


print(chat("My name is Hasnain Malik. I am learning AI using Python."))
print(chat("I am also a Web developer."))
print(chat("What should you remember about me?"))