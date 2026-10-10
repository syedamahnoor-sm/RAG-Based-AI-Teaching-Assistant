import whisper
import json
import os

# Create output directory
os.makedirs("jsons", exist_ok=True)

# Load Whisper model
model = whisper.load_model("large-v2")

# Get audio files
audios = sorted(os.listdir("audios"))

for audio in audios:

    if not audio.lower().endswith(".mp3"):
        continue

    # Extract tutorial number and title
    number, title = os.path.splitext(audio)[0].split("_", 1)

    print(f"Processing Tutorial {number}: {title}")

    # Transcribe or translate audio
    result = model.transcribe(
        audio=os.path.join("audios", audio),
        language="hi",
        task="translate",
        word_timestamps=False,
    )

    # Create timestamped chunks
    chunks = []

    for segment in result["segments"]:
        chunks.append(
            {
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            }
        )

    # Store complete transcription and segments
    chunks_with_metadata = {
        "chunks": chunks,
        "text": result["text"],
    }

    # Save JSON
    output_path = os.path.join("jsons", f"{number}_{title}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)

    print(f"Saved: {output_path}")
