from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

without_rag_template = """
You are an expert in answering questions about NFL football wide receivers.

Here is the question to answer: {question}
"""

with_rag_template = """
You are an expert in answering questions about NFL football wide receivers.

Here is some relevant player data: {player_data}

Here is the question to answer: {question}
"""

# first ask question without RAG data
prompt = ChatPromptTemplate.from_template(without_rag_template)
chain = prompt | model
print("\n\n-------------------------------")
question = input("Ask your question (q to quit): ")
print("\n\n")
result = chain.invoke({"question": question})
print(result)

#now ask the same question with the RAG data
print("\n\nNow answering the same question with RAG data...\n\n")
prompt = ChatPromptTemplate.from_template(with_rag_template)
chain = prompt | model

player_data = retriever.invoke(question)
result = chain.invoke({"player_data": player_data, "question": question})
print(result)
