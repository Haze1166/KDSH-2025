import streamlit as st
import nltk
import csv
from dotenv import load_dotenv
from nltk.corpus import stopwords
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
from typing import Annotated
import os
import time
import logging
import pathway as pw
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List
import pandas as pd

load_dotenv()

# Configure logging (optional, but helpful for debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize nltk stopwords download (do this outside of function)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class Response(BaseModel):
    is_publishable: Annotated[bool, ..., "Whether the research paper is Publishable or not"]

class Conference(BaseModel):
        id: int
        name: str
        description: str
        topics: List[str]
        deadline: str
        domain: str

# Data preprocessing for Google AI Studio
def remove_stopwords_nltk(text):
    if text is None:
        return ""
    filtered_text = [word for word in text.split(" ") if word.lower() not in stop_words]
    return " ".join(filtered_text)


def initialize_llm(api_key):
    """Initializes the language model with the provided API key."""
    try:
        llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key  # Use the provided API key
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None


def load_pdf(path: str):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        content = []
        for doc in docs:
            content.append(doc.page_content)
        context = "".join(content)
        print(len(context.split(" ")))
        return context
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None


def invoke_with_retry(agent, prompt, max_retries=3, initial_delay=60, backoff_factor=2):
    """Invokes the LLM with a retry mechanism."""
    retry_delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = agent.invoke(prompt)
            return response
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {e}")
            if "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    logging.info(f"Rate limit encountered. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= backoff_factor  # Exponential backoff
                else:
                    st.error("Too many retries due to rate limit. Please try again later.") 
                    return None # Return None if all retries failed
            else:
                st.error(f"Error during analysis: {e}")  # For other exceptions
                return None

def prepare_conference_data():
    """Prepares sample conference data."""
    conferences = [
        Conference(
            id=1,
            name="ICML",
            description="International Conference on Machine Learning.",
            topics=["machine learning", "deep learning", "ai"],
            deadline="2024-08-01",
            domain="computer science",
        ),
        Conference(
            id=2,
            name="NeurIPS",
            description="Conference on Neural Information Processing Systems.",
            topics=["neural networks", "ai", "machine learning"],
            deadline="2024-07-15",
            domain="computer science",
        ),
         Conference(
            id=3,
            name="CVPR",
            description="Conference on Computer Vision and Pattern Recognition.",
            topics=["computer vision", "image processing", "ai"],
            deadline="2024-09-01",
            domain="computer science",
        ),
        Conference(
            id=4,
            name="ACL",
            description="Annual Meeting of the Association for Computational Linguistics.",
            topics=["natural language processing", "nlp", "computational linguistics"],
            deadline="2024-07-01",
            domain="linguistics",
        ),
         Conference(
            id=5,
            name="EMNLP",
            description="Conference on Empirical Methods in Natural Language Processing.",
            topics=["natural language processing", "nlp", "machine learning"],
            deadline="2024-06-01",
            domain="linguistics",
        ),
        Conference(
           id=6,
           name="Bioinformatics",
            description="International Conference on Bioinformatics.",
            topics=["bioinformatics", "genomics", "biological data analysis"],
            deadline="2024-10-01",
            domain="biology",
        ),
    ]

    return conferences

def create_conference_table(conference_data):
    """Creates a Pathway table from conference data."""
    return pw.Table(
        [
            pw.column_from_list(
                conference_data,
                lambda c: c.id,
                name="id"
                ),
                pw.column_from_list(
                    conference_data,
                    lambda c: c.name,
                    name = "name"
                    ),
            pw.column_from_list(
                    conference_data,
                    lambda c: c.description,
                    name="description"
                    ),
             pw.column_from_list(
                conference_data,
                lambda c: " ".join(c.topics),
                name="topics"
            ),
              pw.column_from_list(
                conference_data,
                lambda c: c.deadline,
                 name="deadline"
            ),
             pw.column_from_list(
                conference_data,
                lambda c: c.domain,
                 name="domain"
            ),
        ]
    )

@pw.udf
def get_paper_embedding(text:str):
    # Placeholder: Implement text embedding using Google AI Studio model (or other)
    # In reality, you would use the trained model from AI Studio here.
    # This example uses a simple string hash to create a vector.
    hash_val = abs(hash(text)) % 1000 # simple vector
    return np.array([hash_val for _ in range(10)])  

@pw.udf
def calculate_similarity(paper_embedding, conference_embedding):
    # Placeholder: Implement similarity calculation
    # In reality, you would use cosine or dot product similarity

    return cosine_similarity(paper_embedding.reshape(1,-1), conference_embedding.reshape(1,-1))[0][0]

@pw.udf
def build_response(similarity,name,description,topics,deadline,domain):
    return f"Conference: {name}\nDescription: {description}\nTopics: {topics}\nDeadline: {deadline}\nDomain: {domain}\nSimilarity Score: {similarity:.2f}"


def run_rag_pipeline(paper_details, conferences_table, limit=5):
    """Runs the RAG pipeline using Pathway."""

    # Embed paper and conference details
    paper_embedding_table = conferences_table.select(text=pw.this.topics).update(embedding= get_paper_embedding(pw.this.text))
    user_paper_embedding_table = pw.Table([pw.column_from_list([paper_details],lambda x:x, name="text")]).update(embedding=get_paper_embedding(pw.this.text))

    # Calculate similarity between paper and conference embeddings.
    ranked_conference_table = pw.join(paper_embedding_table.select(embedding=pw.this.embedding),
        user_paper_embedding_table.select(user_embedding=pw.this.embedding)).select(
        id = paper_embedding_table.id,
        name = paper_embedding_table.name,
        description = paper_embedding_table.description,
        topics = paper_embedding_table.topics,
        deadline = paper_embedding_table.deadline,
        domain = paper_embedding_table.domain,
        similarity = calculate_similarity(pw.this.embedding,pw.this.user_embedding)).sort(by = pw.this.similarity, ascending=False).limit(limit)
    
    # Build response with similarity score and conference details.
    response_table = ranked_conference_table.select(result=build_response(pw.this.similarity,pw.this.name,pw.this.description,pw.this.topics,pw.this.deadline,pw.this.domain))

    # Collect the results.
    results = response_table.all()
    return [row.result for row in results]


def main():
    st.title("Research Paper Conference Recommender")

    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", type="password")

    st.sidebar.header("Research Paper Details")
    paper_details = st.sidebar.text_area("Enter details of your research paper (title, abstract, keywords, domain):")
    
    st.sidebar.header("Upload Research Paper")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file", type=["pdf"]
    )

    if api_key:
            llm = initialize_llm(api_key)

            if llm is None:
                    st.error("Please check your API Key")
                    return
            if uploaded_file is not None:
                with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
            
                user_paper = load_pdf("temp.pdf")

                if user_paper:
                    with st.spinner('Analyzing paper for Publishability...'):
                    # Non-publishable examples (you can load these dynamically if needed)
                        rp1 = load_pdf("R004.pdf")
                        rp3 = load_pdf("R010.pdf")  # Changed to a publishable paper for example

                        prompt = f"""
                            You are an expert Research Paper analyst. Based on the given Examples, classify the user's research paper into 'Publishable' and 'Non Publishable' Paper. Examples are without stopwords.
                            Non-publishable:
                            {remove_stopwords_nltk(rp1)}
                            Publishable:
                            {remove_stopwords_nltk(rp3)}
                            User's research paper:
                            {remove_stopwords_nltk(user_paper)}
                            """

                        agent = llm.with_structured_output(Response)
                        try:
                            response = invoke_with_retry(agent, prompt)  # Call the retry function
                            temp={
                                "filename":uploaded_file.name,
                                "text":str(user_paper)[:15]
                            }
                            if response:
                                if response.is_publishable:
                                    temp["publishable"]=1
                                    st.success("The research paper is classified as *Publishable*.")
                                else:
                                    temp["publishable"]=0
                                    st.warning("The research paper is classified as *Non-Publishable*.")
                                try:
                                    csv_file = 'data.csv'
                                    file_exists = os.path.isfile(csv_file)
                                    with open(csv_file, 'a', newline='\n', encoding='utf-8') as csvfile:
                                        writer = csv.writer(csvfile)
                                        if not file_exists: # Write header only once if file doesn't exist
                                            writer.writerow(["publishable", "filename", "text"])
                                        writer.writerow([temp["publishable"], temp["filename"], temp["text"]])
                                    with open(csv_file, 'rb') as f: 
                                        st.download_button( label="Download CSV File", data=f, file_name='data.csv', mime='text/csv' )
                                except Exception as e:
                                        st.error(f"CSV Error: {e}")
                        except Exception as e:
                             st.error(f"Error during analysis: {e}")

                if paper_details and llm:  # Ensure paper details are not empty
                     with st.spinner('Finding relevant conferences...'):
                        # Setup Pathway table
                        conference_data = prepare_conference_data()
                        conferences_table = create_conference_table(conference_data)

                        # Run the RAG pipeline to find suitable conferences
                        recommendations = run_rag_pipeline(paper_details, conferences_table)
                        if recommendations:
                            st.subheader("Recommended Conferences:")
                            for rec in recommendations:
                                st.write(rec)
                        else:
                            st.info("No recommendations found.")
                
    else:
         st.warning("Please enter your Google Gemini API key in the sidebar to proceed.")

if __name__ == "__main__":
    main()
