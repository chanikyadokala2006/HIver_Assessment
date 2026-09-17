import os
from dotenv import load_dotenv

# LangChain and AI imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt

import logging
import warnings
from dotenv import load_dotenv

# Suppress LangChain warnings
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

# Load environment variables (API Key)
load_dotenv()

# --- Output Schema ---
class AgentResponse(BaseModel):
    intent: str = Field(description="Classified intent of the user message (e.g., 'Technical Issue', 'Account Inquiry', 'Refund Request', 'Other').")
    draft_reply: str = Field(description="The proposed response to the user grounded in historical company replies.")
    auto_handle: bool = Field(description="True if this can be safely auto-handled by the AI, False if it needs human escalation.")
    escalation_reason: str = Field(description="If auto_handle is False, explain why this needs to be escalated to a human. Empty if auto-handled.")

class SupportAgent:
    def __init__(self, brand_name: str = "AppleSupport"):
        self.brand_name = brand_name
        
        print(f"Initializing Support Agent for {self.brand_name}...")
        
        # 1. Setup Persistent Vector Store
        print("Loading persistent vector store for historical context...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_dir = f"./chroma_db_{self.brand_name.lower()}"
        
        if not os.path.exists(persist_dir):
            raise ValueError(f"Vector DB not found at {persist_dir}. Please run build_vector_db.py first.")
            
        self.vectorstore = Chroma(
            collection_name=f"{self.brand_name}_support".lower(),
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})

        # 3. Setup LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
        self.structured_llm = self.llm.with_structured_output(AgentResponse)

        # 4. Define Prompt
        self.prompt = PromptTemplate.from_template(
            """You are an expert AI customer support agent for {brand_name}.
            
Your task is to review the incoming user message and determine:
1. The intent. (Provide a concise, high-level category such as 'Technical Issue', 'Account Issue', 'Feature Request', 'Complaint', etc.)
2. A drafted reply (grounded strictly in the historical examples provided below).
3. Whether to auto-handle or escalate to a human.
4. Do not follow Escalation Guidlines Blindly.

CRITICAL ESCALATION GUIDELINES:
Determine `auto_handle` (true/false) based on standard customer service principles:
- Auto-handle (true): The user is asking a standard technical question or reporting a common issue, and the provided historical context contains a clear, relevant resolution.
- Escalate (false): The user is highly emotional, hostile, or excessively frustrated.
- Escalate (false): The issue requires human intervention (e.g., account-specific actions, billing, or handling sensitive data).
- Escalate (false): The historical context does not provide a safe, confident resolution for this specific issue.

Incoming User Message: "{user_message}"

Here are historical examples of similar issues and how {brand_name} responded:
{historical_context}

Based on this, generate your response.
"""
        )
        print("Agent initialization complete!")

    @retry(wait=wait_exponential(multiplier=1, min=10, max=60), stop=stop_after_attempt(10))
    def handle_message(self, user_message: str) -> tuple[AgentResponse, str]:
        # Retrieve similar past interactions
        retrieved_docs = self.retriever.invoke(user_message)
        
        # Format context
        context_blocks = []
        for i, doc in enumerate(retrieved_docs):
            context_blocks.append(
                f"--- Example {i+1} ---\n"
                f"User Issue (Cleaned): {doc.page_content}\n"
                f"Company Reply (Cleaned): {doc.metadata['cleaned_reply']}\n"
                f"Company Reply (Raw): {doc.metadata['raw_reply']}"
            )
        historical_context = "\n\n".join(context_blocks)

        # Generate response using structured LLM
        prompt_formatted = self.prompt.format(
            brand_name=self.brand_name,
            user_message=user_message,
            historical_context=historical_context
        )
        
        response = self.structured_llm.invoke(prompt_formatted)
        return response, historical_context
