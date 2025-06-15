from pathlib import Path
from typing import Optional, Literal
import modal

DEFAULT_QWEN_ARGS = [
    "-ot",
    ".ffn_.*_exps.=CPU",
    "--jinja",
    "--color",
    "-ngl",
    "99",
    "--threads",
    "48",
    "--temp",
    "0.2",
    "--top-k",
    "20",
    "--top-p",
    "0.9",
    "--min-p",
    "0",
    "-c",
    "16384",
    "--prio",
    "3",
    "-no-cnv"

]
example = """
from manim import *

class OpeningManim(Scene):
    def construct(self):
        title = Tex(r"This is some \LaTeX")
        basel = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(basel, shift=DOWN),
        )
        self.wait()

        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*[FadeOut(obj, shift=DOWN) for obj in basel]),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = Tex("This is a grid", font_size=72)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeIn(grid_title, shift=UP),
            Create(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = Tex(
            r"That was a non-linear function \\ applied to the grid"
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.animate.apply_function(
                lambda p: p
                          + np.array(
                    [
                        np.sin(p[1]),
                        np.sin(p[0]),
                        0,
                    ]
                )
            ),
            run_time=3,
        )
        self.wait()
        self.play(Transform(grid_title, grid_transform_title))
        self.wait()
"""
sys = f"""

Follow these instructions EXACTLY without deviation:  

1. Generate ONLY Python code using the Manim library, reflecting the prompt provided.  
2. The output MUST be clean, correct, complete, and immediately executable—NO commentary, notes, or explanations.  
3. The TOTAL animation duration (sum of all wait times) MUST EXACTLY match the provided audio length.  
4. Every animation MUST feature dynamic, continuous movement—objects should animate smoothly across the screen.  
5. Use ONLY standard Manim objects/methods (e.g., Circle, Square, Text, FadeIn, Write, MoveTo, Rotate, Scale). DO NOT use custom, undefined, or non-standard objects.  
6. Ensure all methods are valid in the current Manim Community version (e.g., use `.move_to(ORIGIN)`, NOT `to_center`).  
7. DO NOT include placeholder text or any code implying image use.  
8. Define every element before use. Use smooth transitions, motion paths, scaling, and color changes for visual appeal.  
9. Use appropriate parameters (e.g., font_size, color, scale) for readability. Keep text concise to fit within the screen.  
10. If multiple texts are added, arrange them vertically or in a stacked manner using .next_to() or .arrange() methods.
11. Format the output EXACTLY as shown below—NO extra text outside the code block.  
12. DO NOT include comments!
13. Use Tex() instead of Text()
14. Instead of "GrowFromLeft" use "write"
    """

def model(
    sys_prompt: Optional[str] = sys,
    prompt: Optional[str] = None,
    model_entrypoint: str = "Qwen2.5/unsloth.Q4_K_M.gguf",
    n_predict: int = -1,  # max number of tokens to predict, -1 is infinite
    args: Optional[str] = None,  # string of arguments to pass to llama.cpp's cli
):
    """Run llama.cpp inference on Modal for phi-4 or deepseek r1."""
    import shlex

    org_name = "Qwen"
    # two sample models: the diminuitive phi-4 and the chonky deepseek r1
    

    parsed_args = DEFAULT_QWEN_ARGS if args is None else shlex.split(args)


    #download_model.remote(repo_id, [model_pattern], revision)
    f = modal.Function.from_name("example-llama-cpp", "llama_cpp_inference")
    # call out to a `.remote` Function on Modal for inference

    prompt =   "<|im_start|>system\n" + sys_prompt + "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"

    result = f.remote(
        model_entrypoint,
        prompt,
        n_predict,
        parsed_args,
        store_output= "qwen",
    )

    print(f"🦙 writing response")

    return result