import os
from logger import logger
from const import MODEL, DOWNLOADS_PATH
from client import client
from datetime import datetime, timedelta

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

def find_recent_pdfs_in_downloads(days: int = 7):
    """
    Find all PDF files in the Downloads folder within the last 'days' days.
    Sort them by the most recent modification (or creation) time — the newest file first.
    Return:
    List[dict] — Each element includes:
    {
        "file_name": str,
        "full_path": str,
        "modified_time": datetime
    }
    """
    recent_files = []
    now = datetime.now()
    cutoff_time = now - timedelta(days=days)

    for root, _, files in os.walk(DOWNLOADS_PATH):
        for f in files:
            if f.lower().endswith(".pdf"):
                file_path = os.path.join(root, f)
                try:
                    modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if modified_time >= cutoff_time:
                        recent_files.append({
                            "file_name": f,
                            "full_path": file_path,
                            "modified_time": modified_time
                        })
                except Exception as e:
                    print(f"Error while accessing {file_path}: {e}")
    recent_files.sort(key=lambda x: x["modified_time"], reverse=True)

    return recent_files

recent_pdfs = find_recent_pdfs_in_downloads()
if not recent_pdfs:
    print("Không có file PDF nào được tải xuống trong 7 ngày qua.")
else:
    print("📄 Các file PDF gần đây:")
    for i, f in enumerate(recent_pdfs, 1):
        print(f"{i}. {f['file_name']} - {f['modified_time'].strftime('%Y-%m-%d %H:%M:%S')}")
