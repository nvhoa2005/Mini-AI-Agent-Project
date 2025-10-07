from class_using import RequestType, Summarize, TTS, AgentResponse
from logger import logger
from const import MODEL, MODEL_TTS, OUTPUT_DIR
from client import client
from utils import get_next_filename, return_text_to_speech

model = MODEL
model_tts = MODEL_TTS

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
                    "'summarize' for text summarization and 'read' for text-to-speech. "
                    "Respond with a JSON object containing 'request_type', 'confidence_score', "
                    "and 'description'."
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
    elif route_result.request_type == "read" and route_result.confidence_score >= 0.7:
        tts = handle_tts(user_input)
        return AgentResponse(
            status="done",
            message="Text converted to speech successfully.",
            audio=tts,
            intent="read"
        )
    else:
        return AgentResponse(
            status="unsupported",
            message="Request type unsupported or confidence too low.",
            intent="unsupported"
        )

