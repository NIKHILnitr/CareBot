import os
import re
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


# =====================================================
# STREAMLIT PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="MediBot",
    page_icon="🩺",
    layout="wide"
)


# =====================================================
# FAISS DATABASE PATH
# =====================================================

DB_FAISS_PATH = "vectorstore/db_faiss"


# =====================================================
# CLEAN TEXT FUNCTION
# =====================================================

def clean_text(text):

    # Remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # Remove noisy OCR text
    text = text.replace(
        "GALE ENCYCLOPEDIA OF MEDICINE",
        ""
    )

    return text.strip()


# =====================================================
# LOAD VECTOR STORE
# =====================================================

@st.cache_resource
def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# =====================================================
# CUSTOM PROMPT
# =====================================================

CUSTOM_PROMPT_TEMPLATE = """
You are a professional AI medical assistant.

Use ONLY the provided medical context to answer the question.

Instructions:
- Write a short, clean, human-like answer.
- Summarize the information naturally.
- Never copy raw text from the context.
- Ignore broken sentences or OCR errors.
- If the answer is unclear, say:
  "I could not find a clear answer in the medical documents."
- Do not mention the context directly.
- Do not hallucinate.

Context:
{context}

Question:
{question}

Clean Helpful Answer:
"""


def set_custom_prompt():

    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


# =====================================================
# LOAD LOCAL HUGGINGFACE MODEL
# =====================================================

@st.cache_resource
def load_llm():

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=256,
        temperature=0.2
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return llm


# =====================================================
# CREATE QA CHAIN
# =====================================================

@st.cache_resource
def create_qa_chain():

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    qa_chain = RetrievalQA.from_chain_type(

        llm=load_llm(),

        chain_type="stuff",

        retriever=retriever,

        return_source_documents=True,

        chain_type_kwargs={
            "prompt": set_custom_prompt()
        }
    )

    return qa_chain


# =====================================================
# MAIN APP
# =====================================================

def main():

    st.title("🩺 MediBot - AI Medical Chatbot")

    st.markdown("---")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Previous Messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    prompt = st.chat_input(
        "Ask your medical question..."
    )

    if prompt:

        # Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        try:

            qa_chain = create_qa_chain()

            with st.spinner(
                "Generating response..."
            ):

                response = qa_chain.invoke({
                    "query": prompt
                })

            result = response["result"]

            source_documents = response.get(
                "source_documents",
                []
            )

            # =====================================================
            # FORMAT SOURCE DOCUMENTS
            # =====================================================

            source_text = ""

            if source_documents:

                source_text += (
                    "\n\n---\n"
                    "## 📚 Source Documents\n"
                )

                for i, doc in enumerate(
                    source_documents,
                    start=1
                ):

                    # Clean document preview
                    preview = clean_text(
                        doc.page_content[:600]
                    )

                    # File name
                    source_path = doc.metadata.get(
                        "source",
                        "Unknown Source"
                    )

                    source = os.path.basename(
                        source_path
                    )

                    # Page number
                    page = doc.metadata.get(
                        "page",
                        "N/A"
                    )

                    # Some loaders start at 0
                    if isinstance(page, int):
                        page += 1

                    source_text += f"""

### 📄 Source {i}

**File:** `{source}`  
**Page:** `{page}`

{preview}

---

"""

            final_response = (
                result + source_text
            )

            # Display Assistant Response
            with st.chat_message(
                "assistant"
            ):
                st.markdown(final_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response
            })

        except Exception as e:

            st.error(
                f"❌ Error: {str(e)}"
            )


# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    main()