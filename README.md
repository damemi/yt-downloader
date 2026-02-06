# YouTube to MP4 (local web app)

Download YouTube videos as MP4 from your browser. Optionally set **start** and **end** timestamps to download only a segment (e.g. from 1:30 to 5:00).

## Requirements

- **Python 3.10+**
- **ffmpeg** (for trimming by timestamp). Install:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: [ffmpeg.org](https://ffmpeg.org/download.html)

## Run locally

1. Create a virtual environment and install dependencies:

   ```bash
   cd youtube-downloader
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   python app.py
   ```

3. Open **http://127.0.0.1:5001** in your browser.

4. Paste a YouTube URL, optionally set start/end times (e.g. `0`, `1:30`, `5:00`), and click **Download MP4**.

## Timestamp format

- **Seconds**: `90`
- **Minutes:Seconds**: `1:30`
- **Hours:Minutes:Seconds**: `1:30:00`

Leave **Start** at 0 (or empty) to begin at the start of the video. Leave **End** empty to download to the end of the video.
