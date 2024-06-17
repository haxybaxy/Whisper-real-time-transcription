import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import cv2
import pygame
import threading

from datetime import datetime, timedelta
from queue import Queue, Empty
from time import sleep

# Set TERM environment variable
os.environ['TERM'] = 'xterm-256color'  # or another appropriate value

# Set CUDA environment variable if using GPU acceleration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def transcribe_audio(data_queue, transcription_queue, device, model, language, phrase_timeout):
    phrase_time = None
    buffer = ""

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    transcription_queue.put(buffer.strip())
                    buffer = ""

                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.tensor(audio_np).to(device)

                result = model.transcribe(audio_tensor, language=language, fp16=device == "cuda")
                text = result['text'].strip()

                buffer += text + " "

                transcription_queue.put(buffer.strip())

            sleep(0.1)
        except Exception as e:
            print(f"Error in transcription thread: {e}")
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=1,
                        help="How real-time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--max_words", default=10,
                        help="Maximum number of words to display at once.", type=int)
    parser.add_argument("--language", default="en",
                        help="Language for the transcription model.", type=str)
    parser.add_argument("--output_file", default="transcription.txt",
                        help="Output file to save the transcription.", type=str)

    args = parser.parse_args()

    data_queue = Queue()
    transcription_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_model = whisper.load_model("small").to(device)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    max_words = args.max_words
    language = args.language
    output_file = args.output_file

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    transcription_thread = threading.Thread(target=transcribe_audio, args=(data_queue, transcription_queue, device, audio_model, language, phrase_timeout))
    transcription_thread.daemon = True
    transcription_thread.start()

    def update_buffer(text, buffer, max_words):
        words = buffer.split()
        words += text.split()
        while len(words) > max_words:
            words.pop(0)
        return ' '.join(words)

    buffer = ""

    # Initialize Pygame
    pygame.init()
    font = pygame.font.SysFont('Arial', 24)
    screen = pygame.display.set_mode((1280, 720))  # Set a larger resolution for the window
    pygame.display.set_caption("Live Feed with Captions")

    # OpenCV video capture
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the frame height

    while True:
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Update buffer from transcription queue
            try:
                while True:
                    new_text = transcription_queue.get_nowait()
                    buffer = update_buffer(new_text, buffer, max_words)
            except Empty:
                pass

            # Clear the screen
            screen.fill((0, 0, 0))

            # Convert the frame to a format suitable for Pygame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = pygame.surfarray.make_surface(frame)

            # Blit the frame onto the screen
            screen.blit(frame, (0, 0))

            # Render the text
            text_surface = font.render(buffer, True, (255, 255, 255))
            screen.blit(text_surface, (10, 680))  # Adjust the position to fit the larger window

            # Update the display
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    return

            sleep(0.1)  # Reduce sleep time to make the feed more responsive
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

    # Save the final transcription to the output file
    with open(output_file, 'a') as f:
        f.write('\n'.join(transcription))

if __name__ == "__main__":
    main()
