import os
import subprocess

files = sorted(os.listdir("videos"))

for tutorial_number, file in enumerate(files, start=1):
    if file.lower().endswith(".mp4"):
        file_name = os.path.splitext(file)[0]

        print(tutorial_number, file_name)
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                f"videos/{file}",
                f"audios/{tutorial_number}_{file_name}.mp3",
            ]
        )
