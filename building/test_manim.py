from manim import *

class BinaryTreeAnimation(Scene):
    def construct(self):
        root = Circle(radius=0.3, color=BLUE, fill_opacity=0.5).move_to(UP*3)
        root_text = Text("1", font_size=36).move_to(root)

        left_child = Circle(radius=0.3, color=GREEN, fill_opacity=0.5).move_to(LEFT*2 + DOWN*1.5)
        left_text = Text("2", font_size=36).move_to(left_child)

        right_child = Circle(radius=0.3, color=GREEN, fill_opacity=0.5).move_to(RIGHT*2 + DOWN*1.5)
        right_text = Text("3", font_size=36).move_to(right_child)

        left_left_child = Circle(radius=0.3, color=RED, fill_opacity=0.5).move_to(LEFT*3 + DOWN*3)
        left_left_text = Text("4", font_size=36).move_to(left_left_child)

        left_right_child = Circle(radius=0.3, color=RED, fill_opacity=0.5).move_to(LEFT + DOWN*3)
        left_right_text = Text("5", font_size=36).move_to(left_right_child)

        right_left_child = Circle(radius=0.3, color=RED, fill_opacity=0.5).move_to(RIGHT + DOWN*3)
        right_left_text = Text("6", font_size=36).move_to(right_left_child)

        right_right_child = Circle(radius=0.3, color=RED, fill_opacity=0.5).move_to(RIGHT*3 + DOWN*3)
        right_right_text = Text("7", font_size=36).move_to(right_right_child)

        line1 = Line(root.get_bottom(), left_child.get_top(), color=WHITE)
        line2 = Line(root.get_bottom(), right_child.get_top(), color=WHITE)
        line3 = Line(left_child.get_bottom(), left_left_child.get_top(), color=WHITE)
        line4 = Line(left_child.get_bottom(), left_right_child.get_top(), color=WHITE)
        line5 = Line(right_child.get_bottom(), right_left_child.get_top(), color=WHITE)
        line6 = Line(right_child.get_bottom(), right_right_child.get_top(), color=WHITE)

        self.play(FadeIn(root), Write(root_text))
        self.wait(1)
        self.play(FadeIn(line1), FadeIn(left_child), Write(left_text))
        self.play(FadeIn(line2), FadeIn(right_child), Write(right_text))
        self.wait(1)
        self.play(FadeIn(line3), FadeIn(left_left_child), Write(left_left_text))
        self.play(FadeIn(line4), FadeIn(left_right_child), Write(left_right_text))
        self.play(FadeIn(line5), FadeIn(right_left_child), Write(right_left_text))
        self.play(FadeIn(line6), FadeIn(right_right_child), Write(right_right_text))
        self.wait(2)