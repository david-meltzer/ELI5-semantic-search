import faiss
import pickle
import numpy as np
import requests
import streamlit as st

from datasets import load_dataset

@st.cache
def read_data(dataset_repo='dhmeltzer/ELI5_embedded'):
    """Read the data from huggingface."""
    return load_dataset(dataset_repo)['train']

@st.cache
def load_faiss_index(path_to_faiss="./faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)

def main():
    # Load data and FAISS index
    data = read_data()
    faiss_index = load_faiss_index()

    # Checkpoint from Huggingface used to encode Reddit posts.
    model_id="sentence-transformers/all-MiniLM-L6-v2"
    
    # Use feature-extraction API to get sentence embeddings.
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    # Token to access Huggingface Inference API.
    headers = {"Authorization": f"Bearer {st.secrets['HF_token']}"}

    def query(texts):
        """
        Encodes the input using a Huggingface transformer model.

        Input:
        -------
        - texts (list): list of strings to encode.

        Output:
        ------
        - list of sentence embedding vectors.
        """
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        return response.json()


    st.title("Semantic Search for Questions on Reddit.")

    st.write("""This application lets you perform a semantic search through questions in the r/ELI5 [dataset](https://huggingface.co/datasets/eli5). \
    The questions and user input are encoded into a high-dimensional vectors space using a Sentence-Transformer model, and in particular the checkpoint [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2").
    To perform the search we use FAISS, which performs an efficient similarity search through the (vectorized) questions.
    The ELI5 dataset contains posts from three subreddits, AskScience (asks), AskHistorians (askh), and ExplainLikeImFive (eli5).
    The score corresponds to the rating each answer recieved when posted on Reddit.
    \n You can use the slider on the left to change the number of results shown.
    We unfortunately cannot verify the veracity of any of the answers posted!""")

    
    # User search with default question.
    user_input = st.text_area("Search box", "What is spacetime made out of?")

    # Filters
    # num_results determines how many results to show the user.
    st.sidebar.markdown("**Filters**")
    num_results = st.sidebar.slider("Number of search results", 1, 50, 5)

    # Vector representation of user input text.
    vector = query([user_input])
    
    # Fetch results
    if user_input:
        # Get IDs for each search result. 
        _, I = faiss_index.search(np.array(vector).astype("float32"), k=num_results)
        
        # Get individual results
        for id_ in I.flatten().tolist():
            row = data[id_]
            
            # List of answers for each reddit post.
            answers=row['answers']['text']
            # URLs listed in each answer.
            answers_URLs = row['answers_urls']['url']
            
            # For each answer in answers, replace the placeholders such as '_URL_0_' with the actual URL.
            for k in range(len(answers_URLs)):
                answers = [answer.replace(f'_URL_{k}_',answers_URLs[k]) for answer in answers]
            
            # for each search result print the title, what split of the dataset it came from,
            # the score of the top answer, and the text of the top answer.
            st.write(f"**Title**: {row['title']}")
            st.write(f"**Split**: {row['split']}")
            st.write(f"**Score**: {row['answers']['score'][0]}")
            st.write(f"**Top Answer**: {answers[0]}")
            st.write("-"*20) 

if __name__ == "__main__":
    main()
