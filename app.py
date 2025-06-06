import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import numpy as np
import gradio as gr

# Load FAISS index and chunks
FAISS_FILE = "hp1_faiss.index"
CHUNKS_FILE = "hp1_chunks.pkl"

index = faiss.read_index(FAISS_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load finetuned GPT2 model
model = GPT2LMHeadModel.from_pretrained("finetuned_hp")
tokenizer = GPT2Tokenizer.from_pretrained("finetuned_hp")
tokenizer.pad_token = tokenizer.eos_token
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#Chatbot
def rag_chat(question, chat_history):
    question_embedding = embedding_model.encode([question]).astype("float32")

    top_k = 3
    distances, indices = index.search(question_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are a Harry Potter expert. You can only answer questions using the information provided in the Context below.
If the answer is not in the Context, say "I don't know."
Context:
{context}
Question: {question}
Answer:"""

    result = generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    answer_text = result[0]["generated_text"]
    answer_only = answer_text.split("Answer:")[-1].strip()

    chat_history.append((question, answer_only))
    return chat_history, chat_history

#Text generator
def simple_text_generation(prompt_text):
    result = generator(
        prompt_text,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    return result[0]["generated_text"]

#Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("<h1>Harry Potter Text Generator and ChatBot</h1>")
    gr.Markdown("Choose Mode: Chatbot or Text Generator")

    mode_selector = gr.Radio(["Text Generator", "Chatbot"], value="Text Generator", label="Mode")

    # Chatbot UI 
    with gr.Group(visible=False) as chatbot_group:
        chatbot_ui = gr.Chatbot()
        msg = gr.Textbox(label="Ask a question...",placeholder="Example: What was magic according to Harry ?")
        clear_chat = gr.Button("Clear Chat")
        chat_state = gr.State([])

        def user_message(user_input, history):
            return rag_chat(user_input, history)

        msg.submit(user_message, [msg, chat_state], [chatbot_ui, chat_state])
        clear_chat.click(lambda: ([], []), None, [chatbot_ui, chat_state])

    #Text Generator UI
    with gr.Group(visible=True) as textgen_group:
        text_prompt = gr.Textbox(label="Enter some text so Finetuned GPT2 can continue it",placeholder="Example: Harry opened the door and")
        generate_btn = gr.Button("Generate Text")
        generated_output = gr.Textbox(label="Generated Output")

        generate_btn.click(simple_text_generation, text_prompt, generated_output)

    #Mode switch logic
    def toggle_mode(mode):
        return (
            gr.update(visible=(mode == "Chatbot")),
            gr.update(visible=(mode == "Text Generator"))
        )

    mode_selector.change(toggle_mode, mode_selector, [chatbot_group, textgen_group])

demo.launch()
