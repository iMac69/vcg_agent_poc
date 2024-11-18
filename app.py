"""
AI Chat Agent for Brand Archetype Classification

This module implements a Streamlit application that interacts with users to
gather insights about their brand. Based on user responses, it classifies the
brand into primary and secondary archetypes using sentiment analysis and
embedding comparisons. The application utilizes pre-trained models for natural
language processing tasks, ensuring all data processing remains within the
client's environment.
"""

import os
import re
import sys

import numpy as np
import streamlit as st
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)

# Load the sentiment analysis model directly
tokenizer_sentiment = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium")
pipe(messages)

# Define tokenizer_dialogue and model_dialogue for the dialogue generator
tokenizer_dialogue = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model_dialogue = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

import warnings

# Correctly use FutureWarning from the warnings module
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="huggingface_hub.file_download"
)

# Initialize the dialogue generation pipeline
dialogue_generator = pipeline(
    "text-generation",
    model=model_dialogue,
    tokenizer=tokenizer_dialogue,
    device=-1,  # Use CPU; set device=0 if using GPU
    max_length=200,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
)


def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The input text.

    Returns:
        float: Sentiment score ranging from -1 (negative) to 1 (positive).
    """
    inputs = tokenizer_sentiment(text, return_tensors='pt', truncation=True)
    outputs = model_sentiment(**inputs)
    scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
    sentiment_score = scores[1] - scores[0]
    return sentiment_score


def load_documents(folder_path):
    """
    Load text documents from the specified folder.
    Args:
        folder_path (str): Path to the folder containing text files.

    Returns:
        dict: A dictionary mapping filenames to their content.
    """
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r',
                      encoding='utf-8') as f:
                documents[filename] = f.read()
    return documents


def chunk_document(text, max_words=1000):
    """
    Split a document into chunks of approximately max_words.

    Args:
        text (str): The input text.
        max_words (int): Maximum number of words per chunk.

    Returns:
        list: A list of text chunks.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    word_count = 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if word_count + words_in_sentence <= max_words:
            current_chunk += " " + sentence
            word_count += words_in_sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            word_count = words_in_sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def prepare_chunks(documents):
    """
    Prepare chunks from the provided documents.
    Args:
        documents (dict): A dictionary of documents.

    Returns:
        list: A list of chunk dictionaries with metadata.
    """
    all_chunks = []
    for archetype_name, text in documents.items():
        chunks = chunk_document(text)
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                'id': f"{archetype_name}_{idx}",
                'text': chunk,
                'metadata': {
                    'archetype': archetype_name.replace('.txt', ''),
                    'chunk_id': idx,
                    'total_chunks': len(chunks)
                }
            }
            all_chunks.append(chunk_data)
    return all_chunks


def index_chunks(chunks, model, index):
    """
    Generate embeddings for each chunk and store them in a local index.
    Args:
        chunks (list): List of chunk dictionaries with 'id', 'text', and
            'metadata'.
        model (SentenceTransformer): The embedding model.
        index (dict): Local index to store embeddings.
    """
    for chunk in chunks:
        embedding = model.encode(chunk['text']).tolist()
        index[chunk['id']] = {
            'embedding': embedding,
            'metadata': chunk['metadata']
        }


def display_introduction():
    """Display the introduction text."""
    introduction_text = """
    ## Welcome!

    Hi there! I'm thrilled to learn more about your brand. 😊

    **Objective:**
    I'll be asking you a series of questions to gather detailed insights about
    your brand. Based on your responses, I'll classify your brand into primary
    and secondary archetypes. This classification will help us craft a
    marketing strategy that truly resonates with your brand's core identity and
    audience.

    Let's dive in!
    """
    st.write(introduction_text)


def display_conclusion(primary, secondary):
    """
    Display the conclusion with the classified archetypes.

    Args:
        primary (str): Primary archetype.
        secondary (str): Secondary archetype.
    """
    conclusion_text = f"""
    ## Thank You!

    I truly appreciate you taking the time to share your insights.
    Based on your responses, your brand aligns with the following archetypes:

    **Primary Archetype:** {primary}
    """
    if secondary:
        conclusion_text += f"\n**Secondary Archetype:** {secondary}"
    conclusion_text += """

    This classification will guide us in crafting a tailored marketing strategy
    that resonates deeply with your audience and aligns perfectly with your
    brand's identity.

    If you have any feedback or would like to discuss these archetypes further,
    feel free to let me know!"""
    st.write(conclusion_text)


def generate_dynamic_response(conversation_history):
    """
    Generate a dynamic response using the local language model.

    Args:
        conversation_history (list): List of conversation turns.

    Returns:
        str: The assistant's generated response.
    """
    # Concatenate the conversation history into a single prompt
    prompt = ''
    for turn in conversation_history:
        role = turn['role']
        content = turn['content']
        if role == 'assistant':
            prompt += f"Assistant: {content}\n"
        else:
            prompt += f"User: {content}\n"
    prompt += "Assistant:"

    # Generate a response using the dialogue generator
    generated = dialogue_generator(prompt, max_length=500, num_return_sequences=1)
    response = generated[0]['generated_text'][len(prompt):].strip()

    # Post-process the response to stop at the first occurrence of a user prompt
    stop_token = 'User:'
    if stop_token in response:
        response = response.split(stop_token)[0].strip()

    return response


def interview_flow(questions, responses_key='responses'):
    """
    Manage the interview flow by presenting questions and capturing responses,
    enhanced with a local language model for dynamic conversation.

    Args:
        questions (list): List of predefined questions.
        responses_key (str): Key to store responses in session state.
    """
    # Initialize session state variables if not present
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []  # Stores the conversation history
        st.session_state['current_question'] = 0  # Tracks current question index
        st.session_state[responses_key] = []  # Stores the user's responses

    current_q_idx = st.session_state['current_question']

    # If all questions have been asked, do not display more questions
    if current_q_idx >= len(questions):
        return

    # Display the conversation so far
    for message in st.session_state['conversation']:
        if message['role'] == 'assistant':
            st.markdown(f"**Agent:** {message['content']}")
        else:
            st.markdown(f"**You:** {message['content']}")

    # If it's the first message or the last message was from the user, the assistant asks a question
    if (
        len(st.session_state['conversation']) == 0
        or st.session_state['conversation'][-1]['role'] == 'user'
    ):
        # Fetch the current question based on the current question index
        current_question = questions[current_q_idx]
        
        # Append the assistant's question to the conversation
        st.session_state['conversation'].append({
            'role': 'assistant',
            'content': current_question
        })

    # User inputs their response
    user_response = st.text_area("Your Answer:", key=f"user_response_{current_q_idx}")

    if st.button("Send", key=f"send_response_{current_q_idx}"):
        if user_response.strip() == "":
            st.warning("Please provide an answer before proceeding.")
        else:
            # Save the user's response
            st.session_state['conversation'].append({'role': 'user', 'content': user_response.strip()})
            st.session_state[responses_key].append(user_response.strip())

            # Generate the agent's next response using the local language model
            with st.spinner("Agent is typing..."):
                assistant_response = generate_dynamic_response(st.session_state['conversation'])

            st.session_state['conversation'].append({'role': 'assistant', 'content': assistant_response})

            # Decide whether to move to the next question or continue the conversation
            if should_move_to_next_question(assistant_response):
                st.session_state['current_question'] += 1


def should_move_to_next_question(assistant_response):
    """
    Determine whether to move to the next question based on the assistant's response.

    Args:
        assistant_response (str): The assistant's generated response.

    Returns:
        bool: True if should move to next question, False otherwise.
    """
    # Simple heuristic based on response length or specific keywords
    if len(assistant_response.split()) < 20 or any(
        phrase in assistant_response.lower()
        for phrase in ["let's move on", "next question", "thank you for sharing", "got it", "understood"]
    ):
        return True
    else:
        return False


def get_archetype_embedding(archetype_name, index):
    """
    Retrieve and average embeddings for a given archetype.

    Args:
        archetype_name (str): The name of the archetype.
        index (dict): The local index containing embeddings.

    Returns:
        list: Averaged embedding vector for the archetype.
    """
    embeddings = [
        entry['embedding']
        for entry in index.values()
        if entry['metadata']['archetype'] == archetype_name
    ]
    if not embeddings:
        return [0.0] * 384  # Return a zero vector if no embeddings found

    # Calculate the average embedding
    archetype_embedding = np.mean(embeddings, axis=0).tolist()
    return archetype_embedding


def classify_archetypes(user_responses, documents, model, index):
    """
    Classify the client into primary and secondary archetypes based on
    responses.

    Args:
        user_responses (list): List of user's responses.
        documents (dict): Dictionary of documents.
        model (SentenceTransformer): The embedding model.
        index (dict): The local index containing embeddings.

    Returns:
        tuple: Primary archetype and secondary archetype (if any).
    """
    # Initialize a dictionary to hold cumulative similarity scores
    archetype_scores = {
        archetype.replace('.txt', ''): 0
        for archetype in documents.keys()
    }

    for response_text in user_responses:
        response_embedding = model.encode(response_text)

        # Analyze sentiment of the response
        sentiment_score = analyze_sentiment(response_text)

        for archetype in archetype_scores:
            archetype_embedding = get_archetype_embedding(archetype, index)
            similarity = cosine(response_embedding, archetype_embedding)
            similarity_score = 1 - similarity
            # Handle cases where cosine similarity might return NaN
            if np.isnan(similarity_score):
                similarity_score = 0
            # Adjust the similarity score by sentiment_score
            adjusted_score = similarity_score * (1 + sentiment_score)
            archetype_scores[archetype] += adjusted_score

    # Sort archetypes based on cumulative scores
    sorted_archetypes = sorted(
        archetype_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    primary_archetype = sorted_archetypes[0][0]
    # Define threshold for secondary archetype (e.g., within 90% of primary score)
    threshold = 0.9 * sorted_archetypes[0][1]

    secondary_archetype = (
        sorted_archetypes[1][0]
        if len(sorted_archetypes) > 1 and sorted_archetypes[1][1] > threshold
        else None
    )

    return primary_archetype, secondary_archetype


def main():
    """Main function to run the Streamlit app."""
    st.title("AI Chat Agent for Brand Archetype Classification")

    # Define the list of predefined questions
    questions = [
        "What's your primary goal in interacting with customers?",
        "How would you describe your ideal brand voice?",
        "What values are most important to your brand?",
        "How do you want customers to feel when interacting with your brand?",
        "What kind of imagery resonates most with your brand identity?",
        "How does your brand approach innovation and change?",
        "What role does tradition play in your brand’s identity?",
        "How does your brand handle adversity or setbacks?",
        "What type of story does your brand most want to tell?",
        "How would your brand approach the concept of luxury and indulgence?",
        "What role does aesthetic beauty play in your brand’s identity?",
        "How does your brand approach moments of celebration and joy?",
        "How does your brand view success and achievement?",
        "What feeling do you think most motivates your customers to take a desired action?",
        "How does your brand handle the unknown and uncertainty?"
    ]

    # Display introduction if not already done
    if 'introduction_displayed' not in st.session_state:
        display_introduction()
        st.session_state['introduction_displayed'] = True

    # Manage the interview flow
    interview_flow(questions)

    # After all questions have been asked
    if st.session_state.get('current_question', 0) >= len(questions):
        st.write("## Processing Your Responses...")

        # Prevent re-processing
        if 'processed' not in st.session_state:
            st.session_state['processed'] = True

            # Load documents
            folder_path = os.path.join(os.getcwd(), 'knowledge_base')
            if not os.path.exists(folder_path):
                st.error(f"Knowledge base directory not found at {folder_path}.")
                st.stop()
            documents = load_documents(folder_path)

            # Prepare chunks
            all_chunks = prepare_chunks(documents)

            # Initialize a local index (dictionary)
            local_index = {}

            # Load the embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Index the chunks
            index_chunks(all_chunks, model, local_index)
            st.success("Indexed all chunks into the local index.")

            # Classify archetypes based on responses
            primary, secondary = classify_archetypes(
                st.session_state['responses'],
                documents,
                model,
                local_index
            )

            # Display classification results
            st.write(f"**Primary Archetype:** {primary}")
            if secondary:
                st.write(f"**Secondary Archetype:** {secondary}")
            else:
                st.write("**No Secondary Archetype Detected.**")

            # Display Conclusion
            display_conclusion(primary, secondary)


if __name__ == "__main__":
    main()
