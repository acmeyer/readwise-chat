from openai import OpenAI
client = OpenAI()
import pandas as pd
import sqlite3
from dataclasses import asdict
from embeddings import (query_embeddings_chroma)
from models import Message
import logging
logging.basicConfig(filename='logs/qa.log', level=logging.INFO)

CHAT_MODEL = "gpt-4o"
MODEL_TEMPERATURE = 0.0

def stream_gpt_response(prompt: str, messages: list[Message]):
    """Returns ChatGPT's response to the given prompt."""
    system_message = [{"role": "system", "content": prompt}]
    if len(messages) > 0:
        message_dicts = [asdict(message) for message in messages]
        conversation_messages = system_message + message_dicts
    else:
        conversation_messages = system_message
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=conversation_messages,
        temperature=MODEL_TEMPERATURE,
        stream=True
    )
    return response

def ask_gpt_chat(prompt: str, messages: list[Message]):
    """Returns ChatGPT's response to the given prompt."""
    system_message = [{"role": "system", "content": prompt}]
    if len(messages) > 0:
        message_dicts = [asdict(message) for message in messages]
        conversation_messages = system_message + message_dicts
    else:
        conversation_messages = system_message
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=conversation_messages,
        temperature=MODEL_TEMPERATURE
    )
    return response.choices[0].message.content

def get_data_for_ids(ids: list) -> pd.DataFrame:
    # Connect to the database
    conn = sqlite3.connect("data/highlights.db")
    c = conn.cursor()
    # Get the data for the given id
    c.execute(f"SELECT id, highlight, book, author, note, location, location_type FROM highlights WHERE id IN ({','.join(ids)})")
    data = c.fetchall()
    df = pd.DataFrame(data, columns=['id', 'highlight', 'book', 'author', 'note', 'location', 'location_type'])
    conn.close()
    return df

def setup_prompt(relevant_docs) -> str:
    """Creates a prompt for gpt-3 for generating a response."""
    formatted_docs = []
    relevant_data = get_data_for_ids(ids=[doc[1] for doc in relevant_docs])
    for _, row in relevant_data.iterrows():
        title = row['book']
        highlight = row['highlight']
        formatted_string = f"Title: {title}\n"
        if pd.notna(row['location_type']) and pd.notna(row['location']):
            location_type = row['location_type']
            location_value = row['location']
            location_string = f"{location_type}: {location_value}"
            formatted_string += f"{location_string}\n"
        formatted_string += f"Highlight: {highlight}\n"
        if pd.notna(row['note']):
            note = row['note']
            note_string = f"My Notes: {note}"
            formatted_string += f"{note_string}\n"
        formatted_docs.append(formatted_string)

    with open('prompt.md') as f:
        prompt = f.read()
        prompt = prompt.replace("$relevant_information", "\n".join(formatted_docs))

    return prompt

if __name__ == "__main__":
    conversation_messages = []
    user_messages = []
    while (user_input := input('You: ').strip()) != "":
        relevant_docs = query_embeddings_chroma(query=user_input, n_results=10)
        prompt = setup_prompt(relevant_docs)
        conversation_messages.append(Message(role="user", content=user_input))
        user_messages.append(Message(role="user", content=user_input))
        answer = stream_gpt_response(prompt, conversation_messages)
        print(f'\nBot: ')
        complete_answer: list[str] = []
        for chunk in answer:
            try:
                event_delta = chunk.choices[0].delta
                answer_text = event_delta.content or ""
                print(answer_text, end='')
                complete_answer.append(answer_text)
            except KeyError:
                pass

        print('\n')
        complete_answer_string = ''.join(complete_answer)
        conversation_messages.append(Message(role="assistant", content=complete_answer_string))