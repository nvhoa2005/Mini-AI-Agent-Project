from typing import Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv
import pathlib
from PyPDF2 import PdfReader
from docx import Document

load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"
model_tts = "gpt-4o-mini-tts"
OUTPUT_DIR = "speech_output"
DOWNLOADS_PATH = str(pathlib.Path.home() / "Downloads")

class RequestType(BaseModel):
    """Router LLM call: Dertermine if user wants to summarize or read text."""
    
    request_type: Literal["summarize", "read raw text", "unsupported", "read file and summary"] = Field(
        description="Type of request being made"
    ),
    confidence_score: float = Field(description="Confidence score between 0 and 1"),
    description: str = Field(description="Cleaned description of the request")
    file_name: Optional[str] = Field(description="Name of the file if applicable")

class Summarize(BaseModel):
    """Response model for text summarization."""
    
    raw_text: str = Field(description="Original text to be summarized")
    summary: str = Field(description="Summarized text")

class TTS(BaseModel):
    """Response model for text-to-speech."""
    
    raw_text: str = Field(description="Original text to be converted to speech")
    audio_content: bytes = Field(description="Audio content in bytes")
    audio_direction: Optional[str] = Field(description="Path to the saved audio file")

class FileContent(BaseModel):
    """Response model for file content reading."""
    
    file_name: str = Field(description="Name of the file intent reading")
    content: str = Field(description="Content of the file")
    summary: Optional[Summarize] = Field(description="Summary of the file content if applicable")

class AgentResponse(BaseModel):
    """Response from the agent: either completed or needs more info."""

    status: Literal["need_input", "done", "unsupported"] = Field(
        description="Status of the agent's response"
    )
    message: str = Field(
        description="Message from the agent"
    )
    audio: Optional[TTS] = Field(
        default=None,
        description="Path to the audio file if text-to-speech was performed"
    )
    summary: Optional[Summarize] = Field(
        default=None,
        description="Summary text if summarization was performed"
    )
    intent: Literal["summarize", "read raw text", "read file and summary", "unsupported"] = Field(
        description="The intent understood by the agent"
    )


def route_request(user_input: str) -> RequestType:
    """Router LLM call to determine if user wants to summarize or read text."""
    logger.info("Routing request...")
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at classifying user requests into two categories: "
                    "'summarize' for text summarization, 'read raw text' for text-to-speech, 'read file and summary' for reading a file and summarizing its content, "
                    "Respond with a JSON object containing 'request_type', 'confidence_score', "
                    ", 'description' and 'file_name' if user request."
                ),
            },
            {
                "role": "user",
                "content": f"Classify the following request: {user_input}",
            },
        ],
        response_format=RequestType,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Request routed as: {result.request_type} with confidence: {result.confidence_score}"
    )
    return result

# Input for Sumarization
user_input1 = """
    Summarize the following text in about 50 words:
    Technology has significantly changed the way people communicate and work. 
    In the past, letters and face-to-face meetings were the main forms of interaction, 
    but now emails, video calls, and instant messaging have made communication 
    faster and more convenient. Businesses rely heavily on technology to increase productivity, 
    manage data, and connect with customers around the world. 
    However, this dependence also brings challenges such as cybersecurity threats and reduced human interaction. 
    As technology continues to evolve, people must learn to balance efficiency with emotional connection 
    to ensure a healthy relationship between humans and machines.
"""
route_request1 = route_request(user_input1)
print(route_request1.request_type, route_request1.confidence_score, route_request1.description)

# Input for Text-to-Speech
user_input2 = """
    Read the following text for me:
    Technology has significantly changed the way people communicate and work. 
    In the past, letters and face-to-face meetings were the main forms of interaction, 
    but now emails, video calls, and instant messaging have made communication 
    faster and more convenient. Businesses rely heavily on technology to increase productivity, 
    manage data, and connect with customers around the world. 
    However, this dependence also brings challenges such as cybersecurity threats and reduced human interaction. 
    As technology continues to evolve, people must learn to balance efficiency with emotional connection 
    to ensure a healthy relationship between humans and machines.
"""
route_request2 = route_request(user_input2)
print(route_request2.request_type, route_request2.confidence_score, route_request2.description)

# Unsupported Input
user_input3 = """
    What's the weather like today?
"""
route_request3 = route_request(user_input3)
print(route_request3.request_type, route_request3.confidence_score, route_request3.description)

# Input for Read file and summary
user_input4 = """
    Please read the file name test.pdf and provide a summary in about 50 words of its contents.
"""
route_request4 = route_request(user_input4)
print(route_request4.request_type, route_request4.confidence_score, route_request4.description, route_request4.file_name)


def handle_summarization(text: str, max_words: int = 50) -> Summarize:
    """Handle text summarization."""
    logger.info("Handling summarization...")
    
    prompt = (
        f"""Summarize the following text in about {max_words} words:\n\n{text}
        User input might be had this format:
        - Requested for summarizing text
        - Text to be summarized under the request
        So please summarize the text only, without including the request part.
        """
    )
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": prompt},
        ],
        response_format=Summarize,
    )
    
    summary = completion.choices[0].message.parsed
    logger.info("Summarization completed.")
    
    return summary

# Example usage of handle_summarization
text = """
    Technology has significantly changed the way people communicate and work. 
    In the past, letters and face-to-face meetings were the main forms of interaction, 
    but now emails, video calls, and instant messaging have made communication 
    faster and more convenient. Businesses rely heavily on technology to increase productivity, 
    manage data, and connect with customers around the world. 
    However, this dependence also brings challenges such as cybersecurity threats and reduced human interaction. 
    As technology continues to evolve, people must learn to balance efficiency with emotional connection 
    to ensure a healthy relationship between humans and machines.
"""

summary_result = handle_summarization(text, max_words=50)
print("Original Text:", summary_result.raw_text)
print("Summary:", summary_result.summary)


def get_next_filename(output_dir: str) -> str:
    """
    Check the output directory for existing audio files and determine the next available filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Make dir: {output_dir}")

    # Lọc file .mp3 và lấy phần số
    existing_files = [
        f for f in os.listdir(output_dir)
        if f.endswith(".mp3")
    ]

    if existing_files:
        max_num = max(int(f[:-4]) for f in existing_files)
    else:
        max_num = 0

    next_num = max_num + 1
    next_filename = os.path.join(output_dir, f"{next_num}.mp3")
    return next_filename

def return_text_to_speech(text: str) -> str:
    """Remove the request part from the text."""
    logger.info("Extracting text to be converted to speech...")
    
    prompt = (
        f"""Remove any request part and return only the text to be read from the following input:\n\n{text}"""
    )
    completion = client.beta.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that removes the request parts."},
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content.strip()

# Example usage of return_text_to_speech
text_to_read = """
    Read the following text for me:
    Technology has significantly changed the way people communicate and work. 
"""
cleaned_text = return_text_to_speech(text_to_read)
print("Cleaned Text:", cleaned_text)

def handle_tts(text: str) -> TTS:
    """Handle text-to-speech conversion."""
    logger.info("Starting text-to-speech conversion...")

    # Clean the text to be read
    cleaned_text = return_text_to_speech(text)
    # Call the OpenAI Audio API for TTS
    with client.audio.speech.with_streaming_response.create(
        model=model_tts,
        voice="alloy",
        input=cleaned_text
    ) as response:
        audio_bytes = response.read()
    # Find next available filename
    output_path = get_next_filename(OUTPUT_DIR)
    # Save audio bytes to file
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    logger.info(f"Saved file in: {output_path}")
    return TTS(raw_text=cleaned_text, audio_content=audio_bytes, audio_direction=output_path)

# Example usage of handle_tts
text_to_read = """
    Hello, this is a text-to-speech conversion example using OpenAI's API.
"""
tts_result = handle_tts(text_to_read)
print("Text to Read:", tts_result.raw_text)
print("Audio saved at:", tts_result.audio_direction)

def find_file_in_downloads(filename: str) -> str:
    """
    Search for a file by name in the user's Downloads folder.
    Returns the full path if found, otherwise raises FileNotFoundError.
    """
    for root, _, files in os.walk(DOWNLOADS_PATH):
        for f in files:
            if f.lower() == filename.lower():
                return os.path.join(root, f)
    raise FileNotFoundError(f"File '{filename}' not found in Downloads folder.")

def read_file_content(filepath: str, max_chars: int = 8000) -> str:
    """
    Read text file safely (supports .txt, .md, .csv, .json, .pdf, .docx).
    Truncates if too large to avoid overloading the model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    supported_exts = [".txt", ".md", ".csv", ".json", ".pdf", ".docx"]
    if ext not in supported_exts:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {supported_exts}")

    # --- Handle PDF ---
    if ext == ".pdf":
        text_content = ""
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_content += page_text
                if len(text_content) > max_chars:
                    text_content = text_content[:max_chars]
                    break
        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}")
        return text_content.strip()

    # --- Handle DOCX ---
    if ext == ".docx":
        text_content = ""
        try:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text_content += para.text + "\n"
                if len(text_content) > max_chars:
                    text_content = text_content[:max_chars]
                    break
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {e}")
        return text_content.strip()

    # --- Handle Plain Text Files ---
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read(max_chars)
    return content.strip()

def handle_read_file_and_summary(user_input: str, max_words: int = 50) -> FileContent:
    """Handle reading a file and summarizing its content."""
    logger.info("Handling read file and summary...")

    # Extract file name from user input
    route_result = route_request(user_input)
    if not route_result.file_name:
        raise ValueError("No file name provided in the user input.")

    try:
        filepath = find_file_in_downloads(route_result.file_name)
        print(f"Found file at: {filepath}")
        file_content = read_file_content(filepath)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise

    summary = handle_summarization(file_content, max_words=max_words)

    return FileContent(
        file_name=route_result.file_name,
        content=file_content,
        summary=summary
    )

def process_user_input(user_input: str) -> AgentResponse:
    """Process user input and return an appropriate AgentResponse."""
    route_result = route_request(user_input)
    
    if route_result.request_type == "summarize" and route_result.confidence_score >= 0.7:
        summary = handle_summarization(user_input)
        return AgentResponse(
            status="done",
            message="Text summarized successfully.",
            summary=summary,
            intent="summarize"
        )
    elif route_result.request_type == "read raw text" and route_result.confidence_score >= 0.7:
        tts = handle_tts(user_input)
        return AgentResponse(
            status="done",
            message="Text converted to speech successfully.",
            audio=tts,
            intent="read raw text"
        )
    elif route_result.request_type == "read file and summary" and route_result.confidence_score >= 0.7:
        read_file_and_summary = handle_read_file_and_summary(user_input)
        return AgentResponse(
            status="done",
            message="File read and summarized successfully.",
            summary=read_file_and_summary.summary,
            intent="read file and summary"
        )
    else:
        return AgentResponse(
            status="unsupported",
            message="Request type unsupported or confidence too low.",
            intent="unsupported"
        )

# Example usage of process_user_input
# Input for Sumarization
user_input1 = """
    Summarize the following text in about 50 words:
    Technology has significantly changed the way people communicate and work. 
    In the past, letters and face-to-face meetings were the main forms of interaction, 
    but now emails, video calls, and instant messaging have made communication 
    faster and more convenient. Businesses rely heavily on technology to increase productivity, 
    manage data, and connect with customers around the world. 
    However, this dependence also brings challenges such as cybersecurity threats and reduced human interaction. 
    As technology continues to evolve, people must learn to balance efficiency with emotional connection 
    to ensure a healthy relationship between humans and machines.
"""
agentResponse1 = process_user_input(user_input1)
print(agentResponse1.message)
if agentResponse1.summary:
    print("Summary:", agentResponse1.summary.summary)
    print("Original Text:", agentResponse1.summary.raw_text)

# Input for Text-to-Speech
user_input2 = """
    Read the following text for me:
    Technology has significantly changed the way people communicate and work.
"""
agentResponse2 = process_user_input(user_input2)
print(agentResponse2.message)
if agentResponse2.audio:
    print("Audio saved at:", agentResponse2.audio.audio_direction)
    print("Text to Read:", agentResponse2.audio.raw_text)

# Unsupported Input
user_input3 = """
    What's the weather like today?
"""
agentResponse3 = process_user_input(user_input3)
print(agentResponse3.message)

# Input for Read file and summary
user_input4 = """
    Please read the file name test.docx and provide a summary in about 50 words of its contents.
"""
agentResponse4 = process_user_input(user_input4)
print(agentResponse4.message)
if agentResponse4.summary:
    print("Summary:", agentResponse4.summary.summary)
    print("Original Text:", agentResponse4.summary.raw_text)

