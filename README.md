# 🎬 IMAGIO -  AI-Powered Educational Video Generator

This project is a fully automated AI pipeline that takes any **educational topic** and generates a **narrated animated video** using:

- **Google Gemini (via LangChain)** for content generation
- **Manim** for math-style animations
- **gTTS** for multilingual voiceovers
- **FFmpeg** for audio-video synchronization and merging

---

## 🚀 Features

✅ Converts any topic into multiple structured educational scenes  
✅ Automatically writes Manim animation code using LLMs  
✅ Supports voiceovers in multiple languages  
✅ Automatically renders and synchronizes scenes  
✅ Merges all scenes into a final .mp4 video  
✅ Cleans up intermediate files after processing

---

## 🧠 How It Works

1. **Topic Input:**  
   You specify a topic (e.g., "Binary Tree") and number of scenes.

2. **Scene Script Generation:**  
   Gemini creates clear, structured educational scripts for each scene.

3. **Animation Description:**  
   Gemini also provides a **Manim-based animation description** for each script.

4. **Code Generation:**  
   Manim code is generated using a local LLM (Qwen2.5 via `Unsloth`).

5. **Voiceover (optional):**  
   Scripts are translated and converted to audio using `gTTS`.

6. **Rendering + Sync:**  
   - Manim renders the animation.  
   - FFmpeg pads/fixes durations and muxes audio+video.

7. **Final Merge:**  
   All scene videos are merged into a final `final_video.mp4`.

---

## 📁 Directory Structure

```

output/
├── audio/          # Scene voiceovers (.mp3)
├── scenes/         # Generated Manim Python files
├── videos/         # Rendered & merged video clips (.mp4)
├── media/          # Temp files created by Manim
└── final\_video.mp4 # Final educational video

````

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
````

Required tools & libraries:

* `manim`
* `ffmpeg`
* `gtts`
* `langchain`
* `translatepy`
* `google-generativeai` (Gemini)
* `Unsloth` + `Qwen2.5` LLM

Make sure FFmpeg and Manim are available in your PATH.

---

## 🔑 Google API Key

The project uses **Google Gemini** via LangChain.
Set your API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-google-api-key"
```

Or the script will use a built-in fallback key.

---

## 🧪 Usage

Inside your Python script or notebook:

```python
from Animator import main_workflow

main_workflow(
    topic="Binary Tree",
    language="en",       # Voiceover language (e.g., 'en', 'hi', 'es', 'ar')
    voice=True,          # Set to False to disable voiceover
    no_scenes=4          # Number of scenes to generate
)
```

The final output video will be saved as:

```
output/final_video.mp4
```

---

## 🌐 Supported Languages

Voiceovers can be generated in many supported languages:

* English (`en`)
* Hindi (`hi`)
* Spanish (`es`)
* Arabic (`ar`)
* French (`fr`)
* and many more...

---

## 🧹 Cleanup

Temporary files are auto-deleted after the video is successfully created.

---

## 📦 Future Improvements

* Web-based GUI for input/output
* Custom voiceover via ElevenLabs or Azure TTS
* Scene preview editor
* Subtitle generation

---

## 🙌 Acknowledgements

* [Manim](https://www.manim.community/)
* [LangChain](https://www.langchain.com/)
* [gTTS](https://pypi.org/project/gTTS/)
* [Google Gemini](https://ai.google.dev/)
* [FFmpeg](https://ffmpeg.org/)
* [Unsloth](https://unsloth.ai/) + [Qwen2.5](https://huggingface.co/Qwen)

---

## 📸 Demo Output

> *“Explaining Binary Trees in 4 scenes with voiceover in English”*
> ✅ Animations
> ✅ AI scripts
> ✅ Voice
> ✅ Synced and merged

---

## 📬 Contact

For issues or contributions, feel free to raise a pull request or open an issue!

```

Let me know if you'd like this in HTML format for a website or a shorter version for a GitHub description.
```
