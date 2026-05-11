import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from huggingface_hub import InferenceClient

from langchain_core.language_models.llms import LLM

from pydantic import Field, root_validator

from langchain_core.prompts import PromptTemplate

from langchain_classic.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in .env file")


# ==========================================
# MODEL CONFIG
# ==========================================

HUGGINGFACE_REPO_ID = "google/flan-t5-large"


# ==========================================
# CUSTOM HUGGINGFACE LLM
# ==========================================

class HuggingFaceInference(LLM):

    repo_id: str
    hf_token: Optional[str] = None

    temperature: float = 0.2
    max_new_tokens: int = 512
    top_p: float = 0.95

    client: Any = None

    stop_sequences: List[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }

    @root_validator(pre=True)
    def build_client(cls, values):

        token = values.get("hf_token")

        values["client"] = InferenceClient(
            model=values["repo_id"],
            token=token,
        )

        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:

        response = self.client.text_generation(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return str(response)


# ==========================================
# LOAD LLM
# ==========================================

def load_llm():

    llm = HuggingFaceInference(
        repo_id=HUGGINGFACE_REPO_ID,
        hf_token=HF_TOKEN,
    )

    return llm


# ==========================================
# IMPROVED PROMPT
# ==========================================

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful AI medical assistant.

Answer the user's question using only the provided medical context.

Rules:
- Give a clean and natural answer.
- Summarize information properly.
- Do NOT copy raw context text.
- Do NOT include incomplete sentences.
- If the answer is unclear, say:
  "I could not find a clear answer in the medical documents."
- Do not make up information.

Context:
{context}

Question:
{question}

Helpful Answer:
"""


def set_custom_prompt():

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return prompt


# ==========================================
# LOAD FAISS DATABASE
# ==========================================

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully")


# ==========================================
# CREATE QA CHAIN
# ==========================================

qa_chain = RetrievalQA.from_chain_type(

    llm=load_llm(),

    chain_type="stuff",

    retriever=db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    ),

    return_source_documents=True,

    chain_type_kwargs={
        "prompt": set_custom_prompt()
    }
)

print("Medical chatbot ready")


# ==========================================
# ASK QUESTIONS
# ==========================================

if __name__ == "__main__":

    while True:

        user_query = input("\nAsk Question: ")

        if user_query.lower() == "exit":
            break

        response = qa_chain.invoke({
            "query": user_query
        })

        print("\nANSWER:\n")

        print(response["result"])

        print("\nSOURCE DOCUMENTS:\n")

        for i, doc in enumerate(
            response["source_documents"],
            start=1
        ):

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            print(f"\nDocument {i}")
            print(f"Source: {source}")
            print(f"Page: {page}")

            print(doc.page_content[:500])