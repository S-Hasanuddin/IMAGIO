from Animator import main_workflow

dummy_result = main_workflow(
    topic = "Area and Volume of sphere",
    language="gn",
    voice=True,
    no_scenes=4

)
print(f"Video Saved In {dummy_result}")