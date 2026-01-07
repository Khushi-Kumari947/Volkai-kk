import pdfplumber
import requests
import json
import re
from typing import List, Dict


OLLAMA_URL = "http://localhost:11434/api/generate"


MODEL_NAME = "gemma2:2b"

TARGET_SECTIONS = ["Education", "Experience", "Skills"]


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts raw text from a PDF using pdfplumber.
    """
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


def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text.
    """
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    """
    Splits text into manageable chunks for LLMs.
    """
    if len(text) <= chunk_size:
        return [text]

    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def call_ollama(prompt: str) -> str:
    """
    Calls Ollama API.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()
    return response.json().get("response", "")


def safe_json_parse(text: str) -> Dict[str, str]:
    """
    Extracts and parses JSON safely from LLM output.
    """
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            parsed = json.loads(match.group())
            return {
                section: parsed.get(section, "").strip()
                for section in TARGET_SECTIONS
            }
    except Exception:
        pass

    return {section: "" for section in TARGET_SECTIONS}


def extract_sections_llm(text_chunk: str) -> Dict[str, str]:
    prompt = f"""
You are a resume parsing engine.

STRICT RULES:
- Output ONLY valid JSON
- No markdown
- No explanations
- No text outside JSON
- Use empty string if section is missing

JSON FORMAT:
{{
  "Education": "",
  "Experience": "",
  "Skills": ""
}}

Resume text:
{text_chunk}
"""

    response = call_ollama(prompt)

    
    # print("\n----- RAW LLM RESPONSE -----\n")
    # print(response)
    # print("\n----------------------------\n")

    return safe_json_parse(response)



def merge_results(results: List[Dict[str, str]]) -> Dict[str, str]:
    merged = {section: [] for section in TARGET_SECTIONS}

    for res in results:
        for section in TARGET_SECTIONS:
            if res.get(section):
                merged[section].append(res[section])

    return {
        section: "\n".join(content).strip()
        for section, content in merged.items()
    }


def extract_resume(pdf_path: str) -> Dict[str, str]:
    raw_text = extract_pdf_text(pdf_path)

    if not raw_text:
        raise ValueError("No text extracted from PDF")

    chunks = chunk_text(raw_text)

    llm_outputs = []
    for chunk in chunks:
        llm_outputs.append(extract_sections_llm(chunk))

    return merge_results(llm_outputs)



if __name__ == "__main__":
    # pdf_path = "data/Naukri_BharatRohella[2y_0m].pdf"
    # pdf_path = "data/Naukri_MohdRakeebAhmad[2y_0m].pdf"
    # pdf_path = "data/Naukri_RAGINISAH[2y_1m] (1).pdf"
    pdf_path = "data/Naukri_TulikaSrivastava[Fresher].pdf"
    

    resume_data = extract_resume(pdf_path)

    for section, content in resume_data.items():
        print(f"\n {section} \n")
        print(content if content else "[NOT FOUND]")
