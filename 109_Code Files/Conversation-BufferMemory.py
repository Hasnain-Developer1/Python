from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

memory = ConversationBufferMemory(return_messages=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot. Remember the user's preferences."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

def load_history(_):
    return memory.load_memory_variables({})["history"]

chain = (
    {
        "history": load_history,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)

msg1 = "My name is Hasnain Malik and I am a Web Developer."

res1 = chain.invoke(msg1)

text1 = res1.content[0]["text"]

memory.save_context(
    {"input": msg1},
    {"output": text1}
)

print(text1)

msg2 = "What do you know about me?"

res2 = chain.invoke(msg2)

text2 = res2.content[0]["text"]

memory.save_context(
    {"input": msg2},
    {"output": text2}
)

print(text2)