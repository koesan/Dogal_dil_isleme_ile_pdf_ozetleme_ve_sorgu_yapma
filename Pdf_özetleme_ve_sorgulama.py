from langchain.chains.summarize import load_summarize_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.llms import OpenAI
import openai
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import os
import time

# OpenAI API anahtarınızı ayarlayın
os.environ['OPENAI_API_KEY'] = "sk-"
llm = OpenAI(temperature=0.6)

tr_to_en = "Helsinki-NLP/opus-mt-tr-en"
en_to_tr = "Helsinki-NLP/opus-mt-tc-big-en-tr"

selected_pdf_path = None

# PDF dosyasını şeçme
def open_file():
    global selected_pdf_path
    selected_pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

# Metinleri tercüme etme
def translators(text, model_name):
    translator = pipeline(task="translation", model=model_name)
    output = translator(text, clean_up_tokenization_spaces=True)
    return output

# PDF'yi özetleme
def summarize_pdf(pdf_folder):
    loader = PyPDFLoader(pdf_folder)
    docs = loader.load_and_split()
    chains = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chains.run(docs)
    summary = translators(summary, en_to_tr)
    try:
        return f"Summary:\n {summary[0]['translation_text']}\n"
    except openai.RateLimitError as e:
        return "Kullanım sınırına ulaşıldı. 20 Saniye bekle!"
    except Exception as e:
        return f"Error: {e}"

# Sorgu fonksiyonu 
def query_pdf(pdf_folder, user_query):
    loader = PyPDFLoader(pdf_folder)
    index = VectorstoreIndexCreator().from_loaders([loader])
    try:
        result = index.query(user_query)
        return f"Query Result:\n {result}"
    except openai.RateLimitError as e:
        return "Kullanım sınırına ulaşıldı. 20 Saniye bekle!"
    except Exception as e:
        return f"Error: {e}"

def query_callback():
    global selected_pdf_path
    if selected_pdf_path is None:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Lütfen önce bir PDF dosyası seçin.")
        return

    user_query = query_entry.get()
    if not user_query:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Lütfen bir soru girin.")
        return

    result_text.delete(1.0, tk.END)  
    query_result = query_pdf(selected_pdf_path, user_query)
    result_text.insert(tk.END, query_result)

def summarize_callback():
    global selected_pdf_path
    if selected_pdf_path is None:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Lütfen önce bir PDF dosyası seçin.")
        return

    result_text.delete(1.0, tk.END)
    summary_result = summarize_pdf(selected_pdf_path)
    result_text.insert(tk.END, summary_result)

# Tkinter İle arayüz kısmı
root = tk.Tk()
root.title("PDF Summarizer and Query Tool")
root.geometry("500x450")

select_pdf_button = tk.Button(root, text="Select PDF", width=10, command=open_file)
select_pdf_button.place(x=300, y=60)

summarize_button = tk.Button(root, text="Summarize", width=10, command=summarize_callback)
summarize_button.place(x=210, y=60)

# Diğer bileşenleri ekleyin ve mainloop'u başlatın
query_label = tk.Label(root, text="Enter your question:")
query_label.pack()

query_entry = tk.Entry(root, width=50)
query_entry.pack(pady=10)

query_button = tk.Button(root, text="Query", width=10, command=query_callback)
query_button.place(x=120, y=60)

result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
result_text.place(x=5, y=100) 

root.mainloop()
