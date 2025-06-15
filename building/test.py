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
    "0.7",
    "--top-k",
    "20",
    "--top-p",
    "0.8",
    "--min-p",
    "0",
    "-c",
    "16384",
    "--prio",
    "3",
    "-no-cnv"

]

def model(
    sys_prompt: Optional[str] = None,
    prompt: Optional[str] = None,
    model_entrypoint: str = "codeqwen-1_5-7b-chat-q8_0.gguf",
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

if __name__ == "__main__":

    sys_prompt = """
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
```python
from manim import *
import numpy as np

class PolarDemo(Scene):
    def construct(self):
        def polarf(theta):
            return 2 - 2 * np.sin(theta)

        tempo = 1.5
        axes = PolarPlane(radius_max=4).add_coordinates()
        table = MathTable([[r"\theta", r"r=2-2\sin\theta"],
                           [0, 2], [r"\pi/6", 1], [r"\pi/2", 0], [r"\pi", 2], [r"3\pi/2", 4]]).scale(0.7)
        layout = VGroup(axes, table).arrange(RIGHT, buff=LARGE_BUFF)

        self.play(DrawBorderThenFill(axes), FadeIn(table), run_time=3 * tempo)
        self.wait(tempo)

        image = ImageMobject("cardioid_plot.png")
        image.scale(0.5)
        image.to_corner(UR)
        self.play(FadeIn(image), run_time=tempo)
        self.wait(tempo)

        tvals = [0, PI / 6, PI / 2, PI, 3 * PI / 2]
        colors = [BLUE, GREEN, RED, ORANGE, PURPLE]

        for tval, color in zip(tvals, colors):
            r = polarf(tval)
            vec = Arrow(start=axes.polar_to_point(0, 0),
                        end=axes.polar_to_point(r, tval),
                        color=color, buff=0)
            dot = Dot(axes.polar_to_point(r, tval), color=color)
            self.play(Create(vec), run_time=tempo)
            self.wait(0.5 * tempo)
            self.play(FadeIn(dot), FadeOut(vec), run_time=tempo)
            self.wait(0.5 * tempo)

        t = ValueTracker(0)
        dot = always_redraw(lambda: Dot(axes.polar_to_point(polarf(t.get_value()), t.get_value()), color=RED))
        curve = always_redraw(lambda: ParametricFunction(
            lambda u: axes.polar_to_point(polarf(u), u),
            t_range=[0, t.get_value()],
            color=RED,
            stroke_width=6))

        self.play(FadeIn(dot, curve), run_time=tempo)
        self.wait(tempo)
        self.play(t.animate.set_value(2 * PI), run_time=6 * tempo)
        self.wait(tempo)
``` 
"""

    prompt = """
Create an animation using manim code in python to show a binary tree.
"""
    result = model(sys_prompt,prompt,model_entrypoint="Qwen2.5/unsloth.Q4_K_M.gguf",n_predict = 1024)
    print(result)