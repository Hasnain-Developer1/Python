from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

outline_prompt = ChatPromptTemplate.from_template(
    "Create a short outline for a lecture on: {topic}"
)

lecture_prompt = ChatPromptTemplate.from_template(
    "Write a detailed lecture using this outline:\n{outline}"
)

outline_chain = outline_prompt | llm
lecture_chain = lecture_prompt | llm

topic = "LangChain Chains"

outline = outline_chain.invoke({"topic": topic})
print("OUTLINE:\n", outline.content)

lecture = lecture_chain.invoke({"outline": outline.content})
print("\nLECTURE:\n", lecture.content)