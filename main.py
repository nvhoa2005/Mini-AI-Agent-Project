from functions import process_user_input

def main():
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

if __name__ == "__main__":
    main()