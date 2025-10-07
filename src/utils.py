import os
from logger import logger
from const import MODEL
from client import client

model = MODEL

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