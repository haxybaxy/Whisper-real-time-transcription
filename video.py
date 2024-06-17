import cv2
import numpy as np
import whisper
import torch
import moviepy.editor as mp
from moviepy.editor import VideoFileClip

# Load Whisper model
model = whisper.load_model("large")


def transcribe_audio(audio_path, model):
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    segments = result['segments']
    return segments


def draw_text_on_frame(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_color = (255, 255, 255)
    font_thickness = 2
    bg_color = (0, 0, 0)

    # Calculate the width and height of the text box
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_offset_x, text_offset_y = position

    # Make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y),
                  (text_offset_x + text_width, text_offset_y - text_height - baseline))

    # Draw the background rectangle
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    # Put the text on the frame
    cv2.putText(frame, text, (text_offset_x, text_offset_y - baseline), font, font_scale, font_color, font_thickness)


def add_captions_to_video(video_path, segments, temp_video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    current_segment_idx = 0
    num_segments = len(segments)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # current time in seconds

        if current_segment_idx < num_segments and segments[current_segment_idx]['start'] <= current_time <= \
                segments[current_segment_idx]['end']:
            text = segments[current_segment_idx]['text']
            draw_text_on_frame(frame, text, (10, height - 30))

        if current_segment_idx < num_segments and current_time > segments[current_segment_idx]['end']:
            current_segment_idx += 1

        out.write(frame)

    cap.release()
    out.release()


def main(video_path, output_path):
    # Extract audio from video
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)

    # Transcribe audio
    segments = transcribe_audio(audio_path, model)

    # Temporary video path
    temp_video_path = "temp_video.mp4"

    # Add captions to video
    add_captions_to_video(video_path, segments, temp_video_path)

    # Combine the original audio with the captioned video
    video_with_captions = mp.VideoFileClip(temp_video_path)
    final_video = video_with_captions.set_audio(video.audio)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Cleanup temporary files
    import os
    os.remove(temp_video_path)
    os.remove(audio_path)


if __name__ == "__main__":
    video_path = "./speech.mp4"  # Replace with your input video file path
    output_path = "./output_video_with_captions_final12345.mp4"  # Replace with your desired output video file path
    main(video_path, output_path)
