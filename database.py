
#find tutorial here:https://huggingface.co/learn/cookbook/agents

#We first load a knowledge base on which we want to perform RAG: this dataset is a compilation of the documentation pages for many huggingface packages, stored as markdown.
import os
from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import InferenceClientModel, ToolCallingAgent
import json
from smolagents import Tool
from langchain_core.vectorstores import VectorStore
import datasets

from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings




# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face Hub using the token from environment variables
login(token=os.getenv("HF_TOKEN"))

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")



source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) 
    for doc in knowledge_base
][:10]  # <-- LIMIT HERE, BEFORE SPLITTING

docs_processed = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0).split_documents(source_docs)[:5]










docs_processed = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0).split_documents(source_docs)[:10]

embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(documents=docs_processed, embedding=embedding_model)


all_sources = list(set([doc.metadata["source"] for doc in docs_processed]))


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "source": {"type": "string", "description": "", "nullable": True},
        "number_of_documents": {
            "type": "string",
            "description": "the number of documents to retrieve. Stay under 10 to avoid drowning in docs",
         "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, all_sources: str, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.inputs["source"]["description"] = (
            f"The source of the documents to search, as a str representation of a list. Possible values in the list are: {all_sources}. If this argument is not provided, all sources will be searched.".replace(
                "'", "`"
            )
        )

    def forward(self, query: str, source: str = None, number_of_documents=7) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        number_of_documents = int(number_of_documents)

        if source:
            if isinstance(source, str) and "[" not in str(source):  # if the source is not representing a list
                source = [source]
            source = json.loads(str(source).replace("'", '"'))

        docs = self.vectordb.similarity_search(
            query,
            filter=({"source": source} if source else None),
            k=number_of_documents,
        )

        if len(docs) == 0:
            return "No documents found with this filtering. Try removing the source filter."
        return "Retrieved documents:\n\n" + "\n===Document===\n".join([doc.page_content for doc in docs])








model = InferenceClientModel("deepseek-ai/DeepSeek-V4.1-Flash")

retriever_tool = RetrieverTool(vectordb=vectordb, all_sources=all_sources)
agent = ToolCallingAgent(tools=[retriever_tool], model=model, verbosity_level=0)

agent_output = agent.run("Please show me a LORA finetuning script")

print("Final output:")
print(agent_output)