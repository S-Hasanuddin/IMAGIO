# -----------------------------------------------
# Configuration & Initialization
# -----------------------------------------------
import os
import json
import re
import subprocess
import shutil
from gtts import gTTS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import google.generativeai as genai
import tempfile
import re
from langchain_core.messages import SystemMessage, HumanMessage

print("🚀 Starting Animator.py initialization...")

# Create output directory structure
OUTPUT_DIR = os.path.abspath("output")
SCENES_DIR = os.path.join(OUTPUT_DIR, "scenes")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
MEDIA_DIR = os.path.join(OUTPUT_DIR, "media")

print("📁 Creating output directories...")
# Create directories if they don't exist
for directory in [OUTPUT_DIR, SCENES_DIR, AUDIO_DIR, VIDEO_DIR, MEDIA_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"  ✓ Created directory: {directory}")

# Prompt for Google API key if not set
print("🔑 Checking Google API key...")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        print("  ⚠️ No API key found in environment, using default key...")
        GOOGLE_API_KEY = "AIzaSyDD9i0bDWanKAAFfBLwuIbUtxlXck5Ox-U"
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    except Exception as e:
        print(f"  ❌ Error setting Google API key: {e}")
        GOOGLE_API_KEY = input("Enter your Google API Key: ")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("🤖 Configuring Google AI...")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the AI models
print("🧠 Initializing AI models...")
script_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
print("  ✓ AI models initialized successfully")

# System prompt for JSON scene generation
json_system_message = SystemMessage(content="""
You are an advanced AI model with expertise in generating educational content structure.
Your task is to analyze the given topic and generate a JSON structure containing scenes for an educational video.

Each scene should have:
1. "scene_id": A unique identifier (1, 2, 3, etc.)
2. "script": A clear, educational script for the scene


Scene Flow Requirements:
- Create scenes for a complete educational video
- Each scene should build upon previous concepts
- Start with introduction/title scene
- End with summary/conclusion scene
- Middle scenes should break down the concept step-by-step
                                    
Output Example Format:
{
  "scenes": [
    {
      "scene_id": 1,
      "script": "Welcome to our comprehensive guide on binary trees. In this video, we'll explore what binary trees are, how they are structured, and how we use them in computer science applications."
    },
    {
      "scene_id": 2,
      "script": "Let's start with the basics. A binary tree is a data structure where each node has at most two children, commonly referred to as the left and right child. This hierarchical structure is the basis for many algorithms and systems in computer science."
    },
    {
      "scene_id": 3,
      "script": "Each node in a binary tree can branch into two paths: one to the left and one to the right. These clear directions make traversal algorithms efficient and logical."
    },
]

}

Remember:
- Ensure each scene has a clear visual focus
- Consider the educational flow and visual storytelling
- Keep scripts clear, concise, and educational

Make sure the JSON is valid and properly formatted.
keep it simple and minimalistic.
""")


sys_description = SystemMessage(content= """
You are an advanced AI model with expertise in generating educational content structure specifically for Manim animations.
Follow the Instructions perfectly:
1. You will be given a List containing dicts of scene_id and script for that scene.this list is related to a particular topic.
2. You need to create structurted containing scene_id and a animation description corresponding to the script.       
3. Make the animation description simple and in manim based terms.
these animation description are the descriptions of the visual animation that is shown for its corresponding scenes.

Input Example:    
Topic: Binary Tree
[
    {
      "scene_id": 1,
      "script": "Welcome to our comprehensive guide on binary trees. In this video, we'll explore what binary trees are, how they are structured, and how we use them in computer science applications."
    },
    {
      "scene_id": 2,
      "script": "Let's start with the basics. A binary tree is a data structure where each node has at most two children, commonly referred to as the left and right child. This hierarchical structure is the basis for many algorithms and systems in computer science."
    },
    {
      "scene_id": 3,
      "script": "Each node in a binary tree can branch into two paths: one to the left and one to the right. These clear directions make traversal algorithms efficient and logical."
    },
]

Output structured json schema FORMAT:
{
  "scenes": [
    {
      "scene_id": 1,
      "animation_description": "Create a large title text 'Introduction to Binary Tree' centered at the top using BLUE color and large font. Below it, create a subtitle 'A foundational data structure in computer science' in smaller, gray text. Add a decorative underline using a horizontal line that grows from left to right. Finally, create three bullet points that fade in one by one: 'What is a binary tree?', 'How binary trees are structured', and 'Common operations on binary trees'.",
    },
    {
      "scene_id": 2,
      "animation_description": "Clear the screen with FadeOut of all previous elements. Create a centered text 'What is a Binary Tree?' as a question header. Below it, draw a basic binary tree structure using circles as nodes and lines as edges. Start with a single root node, then add left and right child nodes, and expand to two more levels. Use different colors to differentiate parent and child nodes, and label them as Root, Left Child, Right Child, etc. Animate the tree growing top-down, one level at a time.",
    },
    {
      "scene_id": 3,
      "animation_description": "Zoom in slightly on the root and one branch of the tree. Highlight the path from root to a specific leaf. Use arrows or a color change to show the direction from parent to child. Label each connection as 'Left' or 'Right' to reinforce the binary structure.",
    },
]
}
                                
Definitely follow the output schema.
""")
# -----------------------------------------------
# Utility Functions
# -----------------------------------------------
def get_audio_length(audio_path):
    """Returns the duration of an audio file in seconds using ffprobe."""
    print(f"🎵 Getting audio length for: {audio_path}")
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", audio_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        duration = float(result.stdout)
        print(f"  ✓ Audio duration: {duration} seconds")
        return duration
    except Exception as e:
        print(f"  ❌ Error getting audio length: {e}")
        return None

# -----------------------------------------------
# Step 1: Generate JSON Scene Structure from Topic
# -----------------------------------------------
def generate_scene_json(topic,no_scenes):
    """Generate JSON structure with scenes and descriptions."""
    print(f"\n🎬 Generating scene JSON for topic: {topic}")
    
    try:
        print("  🤖 Sending request to AI model...")
        chat_history = []
        chat_history.append(json_system_message)
        chat_history.append(HumanMessage(content=f"Generate a comprehensive educational video structure for the topic: {topic} in ONLY {no_scenes} scenes."))
        result = script_model.invoke(chat_history)
        print("  ✓ Received response from AI model")
        print()
        json = generate_json(result)
        return json
    except Exception as e:
        print(f"  ❌ Error in scene generation: {e}")
        if hasattr(e, 'response'):
            print(f"  Model response: {e.response}")
        return None
def generate_json(result):  
        # Extract JSON from the response
        json_pattern = r"```(?:json)?\n(.*?)\n```"
        match = re.search(json_pattern, result.content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            # Try to find JSON-like structure in the response
            json_str = result.content.strip()
            # Remove any markdown formatting if present
            json_str = re.sub(r'^```json\s*|\s*```$', '', json_str, flags=re.MULTILINE)
        
        try:
            # Parse the JSON
            scenes_data = json.loads(json_str)
            if not isinstance(scenes_data, dict) or 'scenes' not in scenes_data:
                raise ValueError("Invalid JSON structure: missing 'scenes' key")
            
            print(f"  ✓ Successfully parsed JSON with scenes")
            return scenes_data
        except json.JSONDecodeError as e:
            print(f"  ❌ Error parsing JSON: {e}")
            print(f"  Raw JSON string: {json_str}")
            return None
            


# -----------------------------------------------
# Step 2: Generate Manim Code for a Scene
# -----------------------------------------------
def generate_manim_code_from_description(scene_description, scene_id):
    """Generates complete Manim animation code for a scene with validation and retries."""
    print(f"\n🎨 Generating Manim code for Scene {scene_id}")
    try:
        max_attempts = 3
        attempt = 0
        error_feedback = "NONE"
        current_code = ""

        while attempt < max_attempts:
            try:
                attempt += 1
                print(f"  🔄 Attempt {attempt}/{max_attempts}")

                if attempt == 1:
                    # First attempt - generate initial code
                    manim_prompt = f"""
                    Scene Description: {scene_description}
                    Use the class name as Scene{scene_id} for each scene.
                    Dont Include Images or Gifs

                    Generate the Manim code now:
                    """

                    print("  🤖 Sending request to model...")
                    from deploy import model as deploy_main
                    result = deploy_main(
                        prompt=manim_prompt,
                        model_entrypoint="Qwen2.5/unsloth.Q4_K_M.gguf",
                        n_predict=-1
                    )
                    print("  ✓ Received model response")

                    if not result:
                        raise ValueError("No response received from model")

                    print("  📝 Processing generated code...")
                    code_pattern = r"assistant\n([\s\S]*?)\s*\[end of text\]"
                    internal_code_pattern = r"```(?:python)?\n([\s\S]*?)```"
                    match = re.search(code_pattern, result, re.DOTALL)
                    if match:
                        generated_code = match.group(1).strip()
                        generated_code = re.search(internal_code_pattern, generated_code, re.DOTALL).group(1).strip()
                    else:
                        lines = result.strip().split('\n')
                        code_lines = []
                        in_code = False
                        for line in lines:
                            if line.strip().startswith('from manim import'):
                                in_code = True
                            if in_code:
                                code_lines.append(line)
                        generated_code = '\n'.join(code_lines).strip()

                    print("  ✓ Extracted and cleaned code")

                    # Clean up the code
                    generated_code = re.sub(r'^(system|Follow these instructions.*?)$', '', generated_code, flags=re.MULTILINE)
                    generated_code = re.sub(r'\n\s*\n\s*\n', '\n\n', generated_code)
                    generated_code = generated_code.strip()

                    # Replace GrowFromLeft with Write if it exists
                    if 'GrowFromLeft' in generated_code:
                        print("  🔄 Replacing GrowFromLeft with Write...")
                        generated_code = generated_code.replace('GrowFromLeft', 'Write')
                        print("  ✓ Replacement complete")

                    if not generated_code.startswith("from manim import *"):
                        generated_code = f"from manim import *\n\n{generated_code}"

                    current_code = generated_code
                else:
                    # Subsequent attempts - get error correction
                    print(f"  ❌ Error during rendering: {error_feedback}")
                    corrected_code = get_error_correction(current_code, error_feedback)
                    if corrected_code:
                        print("  🔧 Applying Gemini's corrections...")
                        current_code = corrected_code
                        print("Applied Corrected Code")
                    else:
                        print("  ⚠️ No corrections available from Gemini")
                        continue

                # Save current code to file
                filename = os.path.join(SCENES_DIR, f"scene_{scene_id}.py")
                print(f"  💾 Saving code to {filename}")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(current_code)

                print("  ✓ Code saved successfully")

                # Run and validate the code
                print("  🎥 Executing Manim...")
                success, error = execute_manim(scene_id)

                if success:
                    print(f"  ✅ Scene {scene_id} rendered successfully!")
                    return True
                else:
                    error_feedback = error
                    continue

            except Exception as e:
                print(f"  ⚠️ Error during attempt {attempt}: {str(e)}")
                error_feedback = str(e)
                continue

        print(f"  🚨 Max retries reached for Scene {scene_id}")
        return False
    except Exception as e:
        print(f"  ⚠️ Error during attempt {str(e)}")
        return False

def get_error_correction(code: str, error: str) -> str:
    """Get error correction from Gemini model."""
    try:
        print("  🤖 Requesting error correction from Gemini...")
        prompt = f"""
        The following Manim code generated an error:
        
        Error: {error}
        
        Code:
        {code}
        
        Important Manim Guidelines:
        1. Use only built-in Manim classes from 'manim import *'
        2. Common shapes: Circle, Square, Rectangle, Polygon, Line, Arrow
        3. For custom shapes, use Polygon with explicit coordinates
        4. For right-angled triangles, use Polygon with three points
        5. For text, use Tex() or MathTex()
        6. For animations, use Create(), Write(), FadeIn(), etc.
        
        Please provide a corrected version of the code that fixes the error.
        Only provide the corrected Python code, no explanations.
        Make sure to use only built-in Manim classes and proper coordinate systems.
        """
        
        chat_history = []
        chat_history.append(SystemMessage(content="""
        You are an expert at fixing Manim animation code errors. 
        You understand Manim's built-in classes and coordinate system perfectly.
        You know that custom classes should be avoided and instead use Manim's built-in classes.
        For shapes like triangles, use Polygon with explicit coordinates.
        Provide only the corrected code, no explanations."""))
        chat_history.append(HumanMessage(content=prompt))
        
        result = script_model.invoke(chat_history)
        corrected_code = result.content.strip()
        
        # Extract code if it's wrapped in markdown
        code_pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(code_pattern, corrected_code, re.DOTALL)
        if match:
            corrected_code = match.group(1).strip()
        
        print("  ✓ Received correction from Gemini")
        return corrected_code
    except Exception as e:
        print(f"  ⚠️ Error getting correction from Gemini: {e}")
        return None

# -----------------------------------------------
# Step 3: Generate Voiceover for a Scene
# -----------------------------------------------
def generate_voiceover(script_text, scene_id,language):
    """Generate TTS audio from script text."""
    from translatepy import Translator
    print(f"\n🎤 Generating voiceover for Scene {scene_id}")
    # Clean the text for TTS
    cleaned_text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"() ]", "", script_text)
    print(f"  ✓ Cleaned text for TTS")

    print(f"Translating To {language}")
    translator = Translator()
    translated_result = translator.translate(cleaned_text, language)
    translated_text = translated_result.result  # Get translated text
    print(f"Translated Text:\n{translated_text}\n")

    print("  🔊 Converting text to speech...")
    tts = gTTS(text=translated_text, lang=language, slow = False)
    filename = os.path.join(AUDIO_DIR, f"scene_{scene_id}.mp3")
    tts.save(filename)
    print(f"  ✓ Voiceover saved to {filename}")
    return filename

# -----------------------------------------------
# Step 4: Execute Manim Code to Generate Scene Video
# -----------------------------------------------
def execute_manim(scene_id):
    """Execute Manim to generate the scene video."""
    print(f"\n🎥 Executing Manim for Scene {scene_id}")
    scene_file = os.path.join(SCENES_DIR, f"scene_{scene_id}.py")
    print(scene_file)
    
    # Create a temporary directory for Manim output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("  🚀 Running Manim command...")
            process = subprocess.Popen(
                ["manim", "-ql" , "--media_dir", temp_dir, scene_file, "Scene" + str(scene_id)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace'
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"  ❌ Error executing Manim: {stderr}")
                return False, stderr
                
            print("  ✓ Manim execution completed successfully")
            
            # Search for the generated video file in the temporary directory
            generated_video_path = None
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".mp4") and f"Scene{scene_id}" in file:
                        generated_video_path = os.path.join(root, file)
                        break
                if generated_video_path:
                    break
            
            if not generated_video_path:
                print(f"  ❌ Video file not found in temporary directory")
                return False, "Video file not found"
            
            # Create output directory if it doesn't exist
            os.makedirs(VIDEO_DIR, exist_ok=True)
            
            # Define the final output path
            final_video_path = os.path.join(VIDEO_DIR, f"Scene{scene_id}.mp4")
            
            # Move the video to the output directory
            print(f"  📦 Moving video to output directory...")
            shutil.move(generated_video_path, final_video_path)
            print(f"  ✓ Video moved to: {final_video_path}")
            
            return True, None
            
        except Exception as e:
            print(f"  ❌ Error executing Manim: {e}")
            return False, str(e)

# -----------------------------------------------
# Step 5: Synchronize Audio and Video for a Scene
# -----------------------------------------------
def synchronize_scene(scene_id, audio_file, video_file):
    """
    Synchronize audio and video durations using FFmpeg.
    - If video is shorter, freeze last frame.
    - If audio is shorter, pad audio with silence.
    Returns the filename of the merged video.
    """

    print(f"\n🔄 Synchronizing Scene {scene_id}")
    
    def get_duration(file):
        print(f"  ⏱️ Getting duration for {file}")
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", file
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        duration = float(result.stdout)
        print(f"  ✓ Duration: {duration} seconds")
        return duration

    video_duration = get_duration(video_file)
    audio_duration = get_duration(audio_file)
    max_duration = max(video_duration, audio_duration)
    print(f"  📊 Video duration: {video_duration}s, Audio duration: {audio_duration}s")

    # If video is shorter, extend video by freezing last frame
    if video_duration < audio_duration:
        print("  🎬 Extending video duration...")
        freeze_output = os.path.join(VIDEO_DIR, f"frozen_{scene_id}.mp4")
        freeze_cmd = (
            f'ffmpeg -y -i "{video_file}" -vf "tpad=stop_mode=clone:stop_duration={audio_duration - video_duration}" '
            f'-c:a copy "{freeze_output}"'
        )
        subprocess.run(freeze_cmd, shell=True)
        video_file = freeze_output
        print("  ✓ Video extended")

    # If audio is shorter, pad audio with silence
    if audio_duration < video_duration:
        print("  🔊 Padding audio with silence...")
        padded_audio = os.path.join(AUDIO_DIR, f"padded_{scene_id}.mp3")
        pad_cmd = (
            f'ffmpeg -y -i "{audio_file}" -af "apad=pad_dur={video_duration - audio_duration}" '
            f'-t {video_duration} "{padded_audio}"'
        )
        subprocess.run(pad_cmd, shell=True)
        audio_file = padded_audio
        print("  ✓ Audio padded")

    # Merge audio and video
    print("  🔄 Merging audio and video...")
    merged_output = os.path.join(VIDEO_DIR, f"merged_scene_{scene_id}.mp4")
    merge_cmd = (
        f'ffmpeg -y -i "{video_file}" -i "{audio_file}" -c:v copy -c:a aac -shortest "{merged_output}"'
    )
    subprocess.run(merge_cmd, shell=True)
    print(f"  ✓ Merged output saved to {merged_output}")

    # Clean up temp files
    print("  🧹 Cleaning up temporary files...")
    for f in [os.path.join(VIDEO_DIR, f"frozen_{scene_id}.mp4"), 
              os.path.join(AUDIO_DIR, f"padded_{scene_id}.mp3")]:
        if os.path.exists(f):
            os.remove(f)
    print("  ✓ Cleanup complete")

    return merged_output

# -----------------------------------------------
# Step 6: Merge All Scene Videos using FFmpeg
# -----------------------------------------------
def merge_videos_ffmpeg(scene_video_files):
    """Merge video files using FFmpeg instead of MoviePy for better performance."""
    print("\n🎬 Merging all scene videos...")
    # Create a text file listing all video files for FFmpeg
    list_file = os.path.join(OUTPUT_DIR, "video_list.txt")
    print("  📝 Creating video list file...")
    with open(list_file, "w") as f:
        for video_file in scene_video_files:
            # Use absolute paths in the list file
            f.write(f"file '{os.path.abspath(video_file)}'\n")
    print(f"  ✓ Created list with {len(scene_video_files)} videos")

    final_output = os.path.join(OUTPUT_DIR, "final_video.mp4")
    command = f"ffmpeg -y -loglevel error -f concat -safe 0 -i {list_file} -c copy {final_output}"

    try:
        print("  🔄 Running FFmpeg merge...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Videos merged successfully! Output: {final_output}")
            # Clean up the list file
            os.remove(list_file)
            return final_output
        else:
            print(f"  ❌ FFmpeg merge failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"  🚨 Error during FFmpeg merge: {e}")
        return None

# -----------------------------------------------
# Step 7: Cleanup Temporary Files
# -----------------------------------------------
def cleanup_files(scene_count):
    """Clean up temporary files after processing."""
    print("\n🧹 Starting cleanup process...")
    # Remove Manim's media folder if exists
    print("  📁 Removing media directory...")
    shutil.rmtree(MEDIA_DIR, ignore_errors=True)

    # Clean up scene files
    print("  📝 Cleaning up scene files...")
    for i in range(1, scene_count + 1):
        scene_file = os.path.join(SCENES_DIR, f"scene_{i}.py")
        if os.path.exists(scene_file):
            os.remove(scene_file)
            print(f"    ✓ Deleted: {scene_file}")

    # Clean up any remaining files
    list_file = os.path.join(OUTPUT_DIR, "video_list.txt")
    if os.path.exists(list_file):
        os.remove(list_file)
        print("    ✓ Deleted video list file")

    print("✅ Cleanup complete! All temporary files removed.")

def get_manim_video_path(scene_id):
    """Get the path to the video in Manim's media directory."""
    return os.path.join(MEDIA_DIR, "videos", f"scene_{scene_id}", "1080p60", f"Scene{scene_id}.mp4")

def main_workflow(topic,language,voice,no_scenes):
    """Main workflow for generating educational videos."""
    print(f"\n🚀 Starting main workflow for topic: {topic}")
    try:
        # Generate scene structure
        print("\n📝 Generating scene structure...")
        scenes_data = generate_scene_json(topic,no_scenes)
        print(scenes_data)
        if not scenes_data:
            print("❌ Failed to generate scene structure")
            return "Failed to generate scene structure"
        print("NPPPPPPPPPPPPPPPPPPP")
        chat = []
        chat.append(sys_description)
        chat.append(HumanMessage(content=f"Topic:{topic}\n{scenes_data['scenes']}"))
        result = script_model.invoke(chat)
        print("NOOOOOOOOOOOOOOefangsrjgbleruignliserhg")
        visual_data = generate_json(result)
        
        
        # Process each scene
        print("\n🎬 Processing scenes...")
        scene_video_files = []
        for scene,visual in zip(scenes_data["scenes"],visual_data["scenes"]):
            scene_id = scene["scene_id"]
            print(f"\n📺 Processing Scene {scene_id}")
            
            # Generate Manim code
            print("  🎨 Generating Manim code...")
            manim_code = generate_manim_code_from_description(visual["animation_description"], scene_id)
            if not manim_code:
                print(f"  ❌ Failed to generate Manim code for scene {scene_id}")
                return f"Failed to generate Manim code for scene {scene_id}"
            
            # Get the video file path
            video_file = os.path.join(VIDEO_DIR, f"Scene{scene_id}.mp4")
            if not os.path.exists(video_file):
                print(f"  ❌ Video file not found: {video_file}")
                return f"Video file not found: {video_file}"
    
            if voice:
                #Generate voiceover
                print("  🎤 Generating voiceover...")
                audio_file = generate_voiceover(scene["script"], scene_id,language)
                if not audio_file:
                    print(f"  ❌ Failed to generate voiceover for scene {scene_id}")
                    return f"Failed to generate voiceover for scene {scene_id}"
                            
                # Synchronize scene
                print("  🔄 Synchronizing scene...")
                synced_video = synchronize_scene(scene_id, audio_file, video_file)
                if not synced_video:
                    print(f"  ❌ Failed to synchronize scene {scene_id}")
                    return f"Failed to synchronize scene {scene_id}"
                else:
                    print("  🎤 NOOOOOOOOOOOOOOOOOOOOOOOOOOOOO voiceover...")
            else:
                old_name = f'C:/Users/hussa/Desktop/Taha/ContentG/output/videos/Scene{scene_id}.mp4'
                synced_video = f'C:/Users/hussa/Desktop/Taha/ContentG/output/videos/merged_scene_{scene_id}.mp4'

                # Make sure the file exists
                if os.path.exists(old_name):
                    os.rename(old_name, synced_video)
                    print(f"Renamed '{old_name}' to '{synced_video}'")

            
            scene_video_files.append(synced_video)
            print(f"  ✅ Scene {scene_id} completed successfully")
        
        # Merge all scenes
        print("\n🎬 Merging all scenes...")
        final_video = merge_videos_ffmpeg(scene_video_files)
        if not final_video:
            print("❌ Failed to merge video scenes")
            return "Failed to merge video scenes"
        
        # Cleanup temporary files
        print("\n🧹 Cleaning up...")
        cleanup_files(len(scenes_data["scenes"]))
        
        print(f"\n🎉 Workflow completed successfully! Final video: {final_video}")
        return final_video
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"