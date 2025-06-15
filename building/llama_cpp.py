# ---
# args: ["--n-predict", "1024"]
# ---

# # Run large and small language models with llama.cpp (DeepSeek-R1, Phi-4)

# This example demonstrate how to run small (Phi-4) and large (DeepSeek-R1)
# language models on Modal with [`llama.cpp`](https://github.com/ggerganov/llama.cpp).

# By default, this example uses DeepSeek-R1 to produce a "Flappy Bird" game in Python --
# see the video below. The code used in the video is [here](https://gist.github.com/charlesfrye/a3788c61019c32cb7947f4f5b1c04818),
# along with the model's raw outputs.
# Note that getting the game to run required a small bugfix from a human --
# our jobs are still safe, for now.

# <center>
# <a href="https://gist.github.com/charlesfrye/a3788c61019c32cb7947f4f5b1c04818"> <video controls autoplay loop muted> <source src="https://modal-cdn.com/example-flap-py.mp4" type="video/mp4"> </video> </a>
# </center>

from pathlib import Path
from typing import Optional

import modal

# ## What GPU can run DeepSeek-R1? What GPU can run Phi-4?

# Our large model is a real whale:
# [DeepSeek-R1](https://api-docs.deepseek.com/news/news250120),
# which has 671B total parameters and so consumes over 100GB of storage,
# even when [quantized down to one ternary digit (1.58 bits)](https://unsloth.ai/blog/deepseekr1-dynamic)
# per parameter.

# To make sure we have enough room for it and its activations/KV cache,
# we select four L40S GPUs, which together have 192 GB of memory.

# [Phi-4](https://huggingface.co/microsoft/phi-4),
# on the other hand, is a svelte 14B total parameters,
# or roughly 5 GB when quantized down to
# [two bits per parameter](https://huggingface.co/unsloth/phi-4-GGUF).

# That's small enough that it can be comfortably run on a CPU,
# especially for a single-user setup like the one we'll build here.

GPU_CONFIG = "L4:3"  # for DeepSeek-R1, literal `None` for phi-4

# ## Calling a Modal Function from the command line

# To start, we define our `main` function --
# the Python function that we'll run locally to
# trigger our inference to run on Modal's cloud infrastructure.

# This function, like the others that form our inference service
# running on Modal, is part of a Modal [App](https://modal.com/docs/guide/apps).
# Specifically, it is a `local_entrypoint`.
# Any Python code can call Modal Functions remotely,
# but local entrypoints get a command-line interface for free.

app = modal.App("example-llama-cpp")


@app.local_entrypoint()
def main(
    prompt: Optional[str] = None,
    model: str = "qwen",
    n_predict: int = -1,  # max number of tokens to predict, -1 is infinite
    args: Optional[str] = None,  # string of arguments to pass to llama.cpp's cli
):
    """Run llama.cpp inference on Modal for phi-4 or deepseek r1."""
    import shlex

    org_name = "Qwen"
    # two sample models: the diminuitive phi-4 and the chonky deepseek r1

    if model.lower() == "qwen":
        model_name = "CodeQwen1.5-7B-Chat-GGUF"
        quant = "q8_0"
        model_entrypoint_file = (
            f"codeqwen-1_5-7b-chat-q8_0.gguf"
        )
        model_pattern = f"*{quant}*"
        revision = None
        parsed_args = DEFAULT_DEEPSEEK_R1_ARGS if args is None else shlex.split(args)
    else:
        raise ValueError(f"Unknown model {model}")

    repo_id = f"Qwen/CodeQwen1.5-7B-Chat-GGUF"
    #download_model.remote(repo_id, [model_pattern], revision)

    # call out to a `.remote` Function on Modal for inference
    result = llama_cpp_inference.remote(
        model_entrypoint_file,
        prompt,
        n_predict,
        parsed_args,
        store_output=model.lower() == "qwen",
    )
    output_path = Path("/tmp") / f"llama-cpp-{model}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"🦙 writing response to {output_path}")
    output_path.write_text(result)

    return result


# You can trigger inference from the command line with

# ```bash
# modal run llama_cpp.py
# ```

# To try out Phi-4 instead, use the `--model` argument:

# ```bash
# modal run llama_cpp.py --model="phi-4"
# ```

# Note that this will run for up to 30 minutes, which costs ~$5.
# To allow it to proceed even if your local terminal fails,
# add the `--detach` flag after `modal run`.
# See below for details on getting the outputs.

# You can pass prompts with the `--prompt` argument and set the maximum number of tokens
# with the `--n-predict` argument.

# Additional arguments for `llama-cli` are passed as a string like `--args="--foo 1 --bar"`.

# For convenience, we set a number of sensible defaults for DeepSeek-R1,
# following the suggestions by the team at unsloth,
# who [quantized the model to 1.58 bit](https://unsloth.ai/blog/deepseekr1-dynamic).

DEFAULT_DEEPSEEK_R1_ARGS = [
    "--jinja",
    "--color",
    "-ngl",
    "99",
    "--threads",
    "48",
    "--temp",
    "0.7",
    "--top-k",
    "20",
    "--top-p",
    "0.8",
    "--min-p",
    "0",
    "-c",
    "16384",
    "-n",
    "32768",
    "--prio",
    "3",
    "-no-cnv"

]

# ## Compiling llama.cpp with CUDA support

# In order to run inference, we need the model's weights
# and we need code to run inference with those weights.

# [`llama.cpp`](https://github.com/ggerganov/llama.cpp)
# is a no-frills C++ library for running large language models.
# It supports highly-quantized versions of models ideal for running
# single-user language modeling services on CPU or GPU.

# We compile it, with CUDA support, and add it to a Modal
# [container image](https://modal.com/docs/guide/images)
# using the code below.

# For more details on using CUDA on Modal, including why
# we need to use the `nvidia/cuda` registry image in this case
# (hint: it's for the [`nvcc` compiler](https://modal.com/gpu-glossary/host-software/nvcc)),
# see the [Modal guide to using CUDA](https://modal.com/docs/guide/cuda).

LLAMA_CPP_RELEASE = "b4568"
MINUTES = 60

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
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
    .entrypoint([])
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
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=download_image, volumes={cache_dir: model_cache}, timeout=30 * MINUTES
)
def download_model(repo_id, allow_patterns, revision: Optional[str] = None):
    from huggingface_hub import snapshot_download

    print(f"🦙 downloading model from {repo_id} if not present")

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=cache_dir,
        allow_patterns=allow_patterns,
    )

    model_cache.commit()  # ensure other Modal Functions can see our writes before we quit

    print("🦙 model loaded")


# ## Storing model outputs on Modal

# Contemporary large reasoning models are slow --
# for the sample "flappy bird" prompt we provide,
# results are sometimes produced only after several (or even tens of) minutes.

# That makes their outputs worth storing.
# In addition to sending them back to clients,
# like our local command line,
# we'll store the results on a Modal Volume for safe-keeping.

results = modal.Volume.from_name("llamacpp-results", create_if_missing=True)
results_dir = "/root/results"

# You can retrieve the results later in a number of ways.

# You can use the Volume CLI:

# ```bash
# modal volume ls llamacpp-results
# ```

# You can attach the Volume to a Modal `shell`
# to poke around in a familiar terminal environment:

# ```bash
# modal shell --volume llamacpp-results
# # then cd into /mnt
# ```

# Or you can access it from any other Python environment
# by using the same `modal.Volume` call as above to instantiate it:

# ```python
# results = modal.Volume.from_name("llamacpp-results")
# print(dir(results))  # show methods
# ```

# ## Running llama.cpp as a Modal Function

# Now, let's put it all together.

# At the top of our `llama_cpp_inference` function,
# we add an `app.function` decorator to attach all of our infrastructure:

# - the `image` with the dependencies
# - the `volumes` with the weights and where we can put outputs
# - the `gpu` we want, if any

# We also specify a `timeout` after which to cancel the run.

# Inside the function, we call the `llama.cpp` CLI
# with `subprocess.Popen`. This requires a bit of extra ceremony
# because we want to both show the output as we run
# and store the output to save and return to the local caller.
# For details, see the [Addenda section](#addenda) below.

# Alternatively, you might set up an OpenAI-compatible server
# using base `llama.cpp` or its [Python wrapper library](https://github.com/abetlen/llama-cpp-python)
# along with one of [Modal's decorators for web hosting](https://modal.com/docs/guide/webhooks).


@app.function(
    image=image,
    volumes={cache_dir: model_cache, results_dir: results},
    gpu=GPU_CONFIG,
    timeout=30 * MINUTES,
)
def llama_cpp_inference(
    model_entrypoint_file: str,
    prompt: Optional[str] = None,
    n_predict: int = -1,
    args: Optional[list[str]] = None,
    store_output: bool = True,
):
    import subprocess
    from uuid import uuid4

    if prompt is None:
        prompt = DEFAULT_PROMPT  

    if args is None:
        args = []

    if GPU_CONFIG is not None:
        n_gpu_layers = 9999  # all
    else:
        n_gpu_layers = 0

    if store_output:
        result_id = str(uuid4())
        print(f"🦙 running inference with id:{result_id}")

    command = [
        "/llama.cpp/llama-cli",
        "--model",
        f"{cache_dir}/{model_entrypoint_file}",
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--prompt",
        prompt,
        "--n-predict",
        str(n_predict),
    ] + args

    print("🦙 running commmand:", command, sep="\n\t")
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
    )

    stdout, stderr = collect_output(p)

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, command, stdout, stderr)

    if store_output:  # save results to a Modal Volume if requested
        print(f"🦙 saving results for {result_id}")
        result_dir = Path(results_dir) / result_id
        result_dir.mkdir(
            parents=True,
        )
        (result_dir / "out.txt").write_text(stdout)
        (result_dir / "err.txt").write_text(stderr)

    return stdout


# # Addenda

# The remainder of this code is less interesting from the perspective
# of running LLM inference on Modal but necessary for the code to run.

# For example, it includes the default "Flappy Bird in Python" prompt included in
# [unsloth's announcement](https://unsloth.ai/blog/deepseekr1-dynamic)
# of their 1.58 bit quantization of DeepSeek-R1.

DEFAULT_PROMPT = """ Your an expert in python generation , especailly using manim to create proper and good animations.

Subject: Functional Analysis
Topic: Gelfand Theory

Question: How can the Gelfand transform of a Banach algebra be visualized on the unit circle in the complex plane?

Title: Gelfand Transform: Mapping Banach Algebras on the Complex Circle

Narration:
[Introduction: 0:00] Let's embark on a journey into the world of Functional Analysis, where we'll explore the mystical pathways of Banach algebras with the Gelfand transform as our guide. How does this transform relate to the complex unit circle, and why is it so crucial in understanding algebraic structures?

[Key Concept 1: The Banach Algebra 0:30] Imagine a Banach algebra 'A' — a complete vector space with a continuous multiplication operation. Picture this as a three-dimensional landscape of numbers, each with its own unique coordinates and operations.

[Key Concept 2: The Unit Circle 1:00] Now, envision the complex unit circle— a one-dimensional loop in the complex plane defined by $|z| = 1$. What if we could map our entire Banach space onto this simple, elegant structure?

[Key Concept 3: The Gelfand Transform 1:30] Enter the Gelfand transform, a tool that assigns to each element in our Banach algebra a continuous function, preserving algebraic structure while presenting them on the unit circle.

[Key Insight: Relation to Modulus 2:00] The magic of this transform lies in its ability to respect the modulus of each element. That means eigenvalues, norm calculations — all become intuitive, geometrical visualizations on the unit circle.

[Conclusion: 2:30] Finally, by seeing Banach algebras through the lens of the unit circle, we unveil the hidden symmetries and structural beauty of the algebraic universe, bringing clarity to the abstract world of Functional Analysis.

Visual Elements:
- Reveal a dark background with a glowing complex plane grid. As the narration begins, an ethereal, shimmering unit circle fades in, centered in the complex plane. (Timestamp: 0:00-0:15)
- Illustrate a complex Banach space. Visualize this three-dimensional space with a dynamic landscape of numbers connected by glowing lines and vectors, demonstrating its completeness. (Timestamp: 0:30-1:00)
- Zoom into the complex unit circle. Show numbers orbiting around this circle with dynamic vectors projecting their moduli as light rays. (Timestamp: 1:00-1:30)
- Introduce the Gelfand transform effect. Numbers from the Banach space morph into wavy continuous functions that elegantly trace along the circumference of the unit circle. (Timestamp: 1:30-2:00)
- Highlights of how the modulus is preserved. Showcase animated lines and curves on the unit circle, emphasizing eigenvalues and geometric interpretations of norms. End by zooming out to see the entire picture — a harmonious blend of algebraic structure visualized as a vibrant, balanced glowing circle. (Timestamp: 2:00-2:30)

Equations:
- ||a|| = \sup_{z \in \mathbb{C}, |z|=1} |\varphi_{z}(a)|
- \varphi(a \cdot b) = \varphi(a) \cdot \varphi(b)
- \text{Spec}(a) \subset \{z \in \mathbb{C} : |z| = ||a||\}

Visual Style:
A minimalistic, visually engaging style featuring dark backgrounds with glowing lines and shapes. The color palette uses deep blues, purples, and gold to create an ethereal and mathematical aesthetic, illustrating abstract concepts with clarity and elegance.

Generate manim code to create this animation:

this is the format:
python```
(code in here)
```
"""


def stream_output(stream, queue, write_stream):
    """Reads lines from a stream and writes to a queue and a write stream."""
    for line in iter(stream.readline, b""):
        line = line.decode("utf-8", errors="replace")
        write_stream.write(line)
        write_stream.flush()
        queue.put(line)
    stream.close()


def collect_output(process):
    """Collect up the stdout and stderr of a process while still streaming it out."""
    import sys
    from queue import Queue
    from threading import Thread

    stdout_queue = Queue()
    stderr_queue = Queue()

    stdout_thread = Thread(
        target=stream_output, args=(process.stdout, stdout_queue, sys.stdout)
    )
    stderr_thread = Thread(
        target=stream_output, args=(process.stderr, stderr_queue, sys.stderr)
    )
    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()
    process.wait()

    stdout_collected = "".join(stdout_queue.queue)
    stderr_collected = "".join(stderr_queue.queue)

    return stdout_collected, stderr_collected