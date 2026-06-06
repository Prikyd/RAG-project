from langgraph.graph import StateGraph, END
from langchain_mistralai import ChatMistralAI

from langchain_core.messages import HumanMessage, SystemMessage
from config import MISTRAL_API_KEY, LLM_MODEL, TOP_K_RETRIEVAL, SYSTEM_PROMPT
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    sources: List[str]

def build_rag_chain(vectorstore):
    """Build a LangGraph-based RAG pipeline."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )
    llm = ChatMistralAI(
        model=LLM_MODEL,
        api_key= MISTRAL_API_KEY,
        temperature=0.1
    )

    def retrieve_node(state: RAGState) -> RAGState:
        """Node: retrieve relevant chunks."""
        docs = retriever.invoke(state["question"])
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = list(set([
            f"{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}"
            for d in docs
        ]))
        return {**state, "context": context, "sources": sources}

    def generate_node(state: RAGState) -> RAGState:
        """Node: generate answer with Mistral."""
        prompt = SYSTEM_PROMPT.format(
            context=state["context"],
            question=state["question"]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return {**state, "answer": response.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

def ask(chain, question: str, history: list) -> dict:
    """Run a question through the RAG chain."""
    result = chain.invoke({
        "question": question,
        "context": "",
        "answer": "",
        "sources": []
    })
    return result