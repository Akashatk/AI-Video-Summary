import captions
import json

captions.generate_captions()
with open('captions_output.json', 'r') as file:
    data = json.load(file)
    for segment, frames in data.items():
        print(f"Segment: {segment}")
        prompt = '''You are an expert assistant designed to extract keynotes from a frame by frame description of a video. That is divided into segments of 10 seconds each, with each segment having 3 frames per second described in detail.
                The goal is to provide a comprehensive summary of the video's content based on the detailed descriptions of each frame.
                Given the detailed descriptions of each frame, your task is to identify and summarize the main points and themes presented in the video. Focus on capturing the essence of the content, highlighting significant events, actions, or information conveyed through the sequence of frames. 
                Provide a concise summary that encapsulates the overall message and important details from the video.
                The descriptions are in json format:   {"frame": "frame_0001.jpg", "caption": "caption text"}
                Here are the detailed descriptions of each frame: 
                '''+str(frames[0])
        print("Prompt:")    
        print(prompt)
        print()  # blank line for readability

