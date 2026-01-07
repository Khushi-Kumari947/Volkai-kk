import pdfplumber
import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

TARGET_SECTIONS = ["Education", "Experience", "Skills"]



def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(
                x_tolerance=2,
                y_tolerance=2
            )
            if page_text:
                text += page_text + "\n"
    return clean_text(text)



def clean_text(text):
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()



def chunk_text(text, chunk_size=3000):
    if len(text) < 300:
        return [text]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]



def call_dolphin(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]



def extract_sections_llm(text_chunk):
    prompt = f"""
You are an expert resume parser.

From the resume text below, extract these sections even if headings are implicit:
- Education
- Experience
- Skills

Return ONLY valid JSON in this format:
{{
  "Education": "",
  "Experience": "",
  "Skills": ""
}}

Resume text:
{text_chunk}
"""
    response = call_dolphin(prompt)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {section: "" for section in TARGET_SECTIONS}



def merge_results(results):
    merged = {section: [] for section in TARGET_SECTIONS}

    for res in results:
        for section in TARGET_SECTIONS:
            if res.get(section):
                merged[section].append(res[section])

    return {
        section: "\n".join(content).strip()
        for section, content in merged.items()
    }



def extract_resume(pdf_path):
    raw_text = extract_pdf_text(pdf_path)
    chunks = chunk_text(raw_text)

    llm_outputs = []
    for chunk in chunks:
        llm_outputs.append(extract_sections_llm(chunk))

    return merge_results(llm_outputs)



if __name__ == "__main__":
    pdf_path = "Naukri_AayushiKaushik[2y_5m].pdf"
    resume_data = extract_resume(pdf_path)

    for section, content in resume_data.items():
        print(f"\n{'='*10} {section} {'='*10}\n")
        print(content)