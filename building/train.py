from pathlib import Path
from typing import Optional
from datasets import load_dataset, concatenate_datasets, Dataset
import modal
import pandas as pd
from typing import Literal
GPU_CONFIG = "H200:1" 
import os


app = modal.App("example-llama-cpp")


LLAMA_CPP_RELEASE = "b4568"
MINUTES = 60

cuda_version = "11.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(  # this one takes a few minutes!
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

# ## Storing models on Modal

# To make the model weights available on Modal,
# we download them from Hugging Face.

# Modal is serverless, so disks are by default ephemeral.
# To make sure our weights don't disappear between runs,
# which would trigger a long download, we store them in a
# Modal [Volume](https://modal.com/docs/guide/volumes).

# For more on how to use Modal Volumes to store model weights,
# see [this guide](https://modal.com/docs/guide/model-weights).

model_cache = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)
cache_dir = "/root/.cache/llama.cpp"

download_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}",add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .pip_install("unsloth", "torch", "datasets" ,"transformers==4.51.3")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(  # this one takes a few minutes!
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

results = modal.Volume.from_name("llamacpp-results", create_if_missing=True)
results_dir = "/root/results"

@app.function(
    image=download_image,
    volumes={cache_dir: model_cache, results_dir: results},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    timeout=30 * MINUTES,
)
def train(dataset: pd.DataFrame, animation: Literal[None],model_name: str,samples:list):
    from datasets import Dataset
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer, SFTConfig
    import random

    
    

    def generate_conversation(examples):
        problems  = examples["instruction"]
        solutions = examples["output"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            random_number = random.randint(0, len(samples)-1)

            if random.random() < 0.8:  # 80% chance to include system prompt
                conversations.append([
                    {"role": "system",    "content": animation + samples[random_number]},
                    {"role": "user",      "content": problem},
                    {"role": "assistant", "content": solution},
                ])
            else:
                conversations.append([
                    {"role": "user",      "content": problem},
                    {"role": "assistant", "content": solution},
                ])

        return { "conversations": conversations, }
    
    final_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 8192,   # Context length - can be longer, but uses more memory
            load_in_4bit = True,     # 4bit uses much less memory
            load_in_8bit = False,    # A bit more accurate, uses 2x memory
            full_finetuning = False, # We have full finetuning now!
            token = os.environ["HF_TOKEN"],      # use one if using gated models
        )

        conversations = tokenizer.apply_chat_template(
        final_dataset.map(generate_conversation, batched = True)["conversations"],
        tokenize = False,
        )

        data = pd.concat([
        pd.Series(conversations)
        ])

        data.name = "text"
        combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
        combined_dataset = combined_dataset.shuffle(seed = 3407)


        model = FastLanguageModel.get_peft_model(
            model,
            r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,  # Best to choose alpha = rank or rank*2
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,   # We support rank stabilized LoRA
            loftq_config = None,  # And LoftQ
        )

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = combined_dataset,
            eval_dataset = None, # Can set up evaluation!
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4, # Use GA to mimic batch size!
                warmup_steps = 5,
                num_train_epochs = 2, # Set this for 1 full training run.
                max_steps = 30,
                learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "none", # Use this for WandB etc
                save_strategy = "steps",
                save_steps = 30,
                output_dir = "f{cache_dir}/Nexusflow",
            ),
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")



        trainer_stats = trainer.train(resume_from_checkpoint = False)

        model_cache.commit()  
        

        output_path = Path("/tmp") / f"unsloth-train-{model_name}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🦙 writing response to {output_path}")
        output_path.write_text(str(trainer_stats))
        model.save_pretrained_gguf(f"{cache_dir}", tokenizer, quantization_method = "q2_k")
        model.save_pretrained_gguf(f"{cache_dir}", tokenizer, quantization_method = "q4_k_m")
        print("q4_k_m DONNNNNNNNNNE!")
        
        print("q4_k_m DONNNNNNNNNNE!")

        model_cache.commit()  

        print("🦙 model loaded")
    except Exception as e:
        print("❌ An error occurred during model training or saving.")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")   # Optional: for full stack trace
        model_cache.commit()


if __name__ == "__main__":
    from examples import examples
    file_path = r"C:\Users\kalam\Downloads\output.json"

    df = pd.read_json(file_path, orient="records", lines=True)
    print(len(df))

    animation = """
Follow these instructions EXACTLY without deviation:  

1. Generate ONLY Python code using the Manim library, reflecting the prompt provided.  
2. The output MUST be clean, correct, complete, and immediately executable—NO commentary, notes, or explanations.  
3. The TOTAL animation duration (sum of all wait times) MUST EXACTLY match the provided audio length.  
4. Every animation MUST feature dynamic, continuous movement—objects should animate smoothly across the screen.  
5. Use ONLY standard Manim objects/methods (e.g., Circle, Square, Text, FadeIn, Write, MoveTo, Rotate, Scale). DO NOT use custom, undefined, or non-standard objects.  
6. Ensure all methods are valid in the current Manim Community version (e.g., use `.move_to(ORIGIN)`, NOT `to_center`).  
7. IF an image is provided (e.g., Image:cardioid_plot.png), you MUST DEFINITELY use ImageMobject to display it within the animation using appropriate positioning, size and animation (e.g., `FadeIn`, `.scale`).
8. DO NOT include placeholder text or any code implying image use.  
9. Define every element before use. Use smooth transitions, motion paths, scaling, and color changes for visual appeal.  
10. Use appropriate parameters (e.g., font_size, color, scale) for readability. Keep text concise to fit within the screen.  
11. If multiple texts are added, arrange them vertically or in a stacked manner using .next_to() or .arrange() methods.
12. Format the output EXACTLY as shown below—NO extra text outside the code block.  
14. DO NOT include comments!

Example: 

    """
    f = modal.Function.from_name("example-llama-cpp", "train")
    f.remote(
        df, animation, "Nexusflow/Athene-V2-Chat", examples
    )