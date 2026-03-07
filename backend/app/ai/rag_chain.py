# from langchain.chains import RetrievalQA
# from app.ai.llm import get_llm
# from langchain import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain

# # We keep a simple retrieval + LLM chain via LangChain RetrievalQA
# def build_chain(retriever):
#     llm = get_llm()
#     # Using LangChain Retriever + LLM abstraction
#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
#     return qa