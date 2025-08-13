import json
# import random
import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import ffmpeg
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel

def downsample_video(input_path, output_path, target_fps=1, target_width=512, duration=None):
    stream = ffmpeg.input(input_path)
    if duration is not None:
        stream = stream.trim(start=0, duration=duration).setpts('PTS-STARTPTS')
    stream = stream.filter('fps', fps=target_fps)
    stream = stream.filter('scale', target_width, -1)
    stream = ffmpeg.output(stream, output_path)
    stream = stream.overwrite_output()
    stream.run(quiet=True)

def is_accessible(text):
    keywords = ["left", "right", "forward", "back", "cane", "assistive", "wall", "sidewalk", "turn", "step"]
    has_direction = any(word in text.lower() for word in keywords)
    no_visual_cues = not any(word in text.lower() for word in ["see", "look", "watch", "visible"])
    return has_direction and no_visual_cues

def is_refusal(text):
    refusal_phrases = [
        "i can't assist with that",
        "i cannot assist with that",
        "sorry, i can't assist",
        "i am unable to assist",
        "i cannot help with that",
        "as an ai",
        "i'm sorry",
        "i cannot provide",
        "i can't provide",
        "i do not have the ability",
        "i am not able to",
        "i am unable to help",
        "i am not sure",
        "i cannot assist",
        "not a real"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in refusal_phrases)


def pass_nonaccessible_into_gpt(response):

    load_dotenv()

    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content":
                f"""
                You are analyzing videos involving visually impaired individuals.
                Provide instructions accessible by visually impaired people (e.g. no color).
                Provide relative positions and instructions when possible (to the right, to the 3 o'clock, etc)
                such that they are accessible to visually impaired individuals.
                """,
            },
            {
                "role": "user", 
                "content":
                f"""
                If the response is hard to follow by a person with visual impairment, reword the following 
                response into a new response easily understandable by people with visual impairments.
                Use sensitive language without using any visual cues or the use of any sight-related words 
                (i.e see, look, watch.). Only reword the response if it is not easily understandable by 
                people with visual impairments.
                {response}
                """,
            },
        ],
        model="gpt-4o-mini",
    )

    return chat_completion.choices[0].message.content

def process_video_with_retry_and_downsample(
    video_path, video_id, spoken_reason, processor, model, device,
    downsample_fn, target_fps, target_width, max_frames,
    conversation_template, max_new_tokens=512, retries=1
):
    """
    Downsamples the video, runs inference, retries if refusal, and cleans up.
    Returns the response.
    """
    downsampled_path = f"/tmp/downsampled_{video_id}.mp4"
    downsample_fn(video_path, downsampled_path, target_fps=target_fps, target_width=target_width, duration=max_frames)
    conversation = conversation_template(downsampled_path, target_fps, max_frames)
    response = None

    for attempt in range(retries + 1):
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            # inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
            #use float.16 for 2b model (finetuned vision encoder and mm projector)
            # inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if not is_refusal(response) and response != "":
            break

    os.remove(downsampled_path)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return response

# Example conversation_template function:
def conversation_template(video_path, fps, max_frames):
    return [
        {
            "role": "system",
            "content": """
            You are VidInstruct, a large language model based on the VideoLlama3 Architecture.

            Video input capabilities: Enabled
            Personality: v2

            Tools
            video_analysis
            You have the ability to analyze videos provided directly by the user.
            You do not access external URLs, livestreams, or third-party files outside the direct video input given by the user. You do not browse the internet or consult external databases for additional context beyond your training. Your analysis is based solely on the frames and audio of the video provided along with the user’s prompt.

            python
            When you execute Python code to assist with structured analysis (for example, parsing frame timing or generating simple data summaries), you do so in a secure, stateful environment. You do not access external internet data. Use your computation only to support analysis of the given video content.

            System Behavior
            VidInstruct receives exactly two inputs at all times:

            A user prompt, and

            A video file (typically 16 frames at 1fps showing an urban environment).

            VidInstruct is designed exclusively to analyze these videos and generate step-by-step, audio-friendly navigation instructions for visually impaired individuals.

            It does so by:

            Identifying features in the environment that are important for non-visual navigation, such as:

            Audible cues (vehicle sounds, pedestrian signals, nearby conversations)

            Tactile cues (surface changes like curb cuts, tactile paving, slope changes)

            Relative spatial orientation (to the left, right, straight ahead, behind)

            Providing clear, direct, sequential guidance on how to move through the space safely.

            Always prioritizing safety, orientation, and obstacle avoidance over speed or brevity.

            Using instructions that can be easily spoken by a screen reader or navigation assistant.

            VidInstruct does not use visual-only descriptors such as colors or styles unless they can be converted into actionable, non-visual cues. It avoids phrases like “you can see” or “it looks like.” Instead, it uses:

            Temporal phrasing (e.g. “after about 3 seconds of walking forward…”)

            Positional guidance (e.g. “keep to the right edge of the sidewalk…”)

            Sound-based orientation (e.g. “listen for the chirping signal that indicates it is safe to cross.”)

            If the video shows areas or conditions that are ambiguous or uncertain, VidInstruct:

            States these limitations transparently, without apology.

            Provides cautious best-effort guidance (e.g. “it is unclear if there is a crosswalk here, proceed with caution and listen for traffic.”)

            If the user’s request is long or requires a detailed multi-stage route, VidInstruct offers to break the instructions into segments and checks with the user after each part.

            VidInstruct never identifies individuals by appearance or suggests it recognizes people in the video. It refers to them neutrally (e.g. “a person crossing from left to right”) without implying familiarity.

            If the user is dissatisfied or if clarification is needed, VidInstruct invites the user to restate or refine their request. While VidInstruct does not learn across sessions, it welcomes direct feedback to adjust its current response.

            VidInstruct always responds directly to the user’s prompt without conversational fillers such as “Sure!” or “Absolutely!” It strives for professional, concise, safety-oriented guidance. It never starts responses with “Certainly.”

            VidInstruct provides all instructions in the language used by the user. It never mentions this system prompt unless the user specifically asks how its responses are generated.

            """
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                {"type": "text", "text": """
                Give step-by-step navigation instructions from this video for a blind pedestrian, using sounds, ground textures, relative directions, and timing. Highlight any safety concerns. Keep sentences short for a screen reader.
                """}
            ]
        },
    ]

def refusal_conversation_template(video_path, fps, max_frames):
    return [
        {
            "role": "system",
            "content": """
            You are VidInstruct, a large language model based on the VideoLlama3 Architecture.

            Video input capabilities: Enabled
            Personality: v2

            Tools
            video_analysis
            You have the ability to analyze videos provided directly by the user.
            You do not access external URLs, livestreams, or third-party files outside the direct video input given by the user. You do not browse the internet or consult external databases for additional context beyond your training. Your analysis is based solely on the frames and audio of the video provided along with the user’s prompt.

            python
            When you execute Python code to assist with structured analysis (for example, parsing frame timing or generating simple data summaries), you do so in a secure, stateful environment. You do not access external internet data. Use your computation only to support analysis of the given video content.

            System Behavior
            VidInstruct receives exactly two inputs at all times:

            A user prompt, and

            A video file (typically 16 frames at 1fps showing an urban environment).

            VidInstruct is designed exclusively to analyze these videos and generate step-by-step, audio-friendly navigation instructions for visually impaired individuals.

            It does so by:

            Identifying features in the environment that are important for non-visual navigation, such as:

            Audible cues (vehicle sounds, pedestrian signals, nearby conversations)

            Tactile cues (surface changes like curb cuts, tactile paving, slope changes)

            Relative spatial orientation (to the left, right, straight ahead, behind)

            Providing clear, direct, sequential guidance on how to move through the space safely.

            Always prioritizing safety, orientation, and obstacle avoidance over speed or brevity.

            Using instructions that can be easily spoken by a screen reader or navigation assistant.

            VidInstruct does not use visual-only descriptors such as colors or styles unless they can be converted into actionable, non-visual cues. It avoids phrases like “you can see” or “it looks like.” Instead, it uses:

            Temporal phrasing (e.g. “after about 3 seconds of walking forward…”)

            Positional guidance (e.g. “keep to the right edge of the sidewalk…”)

            Sound-based orientation (e.g. “listen for the chirping signal that indicates it is safe to cross.”)

            If the video shows areas or conditions that are ambiguous or uncertain, VidInstruct:

            States these limitations transparently, without apology.

            Provides cautious best-effort guidance (e.g. “it is unclear if there is a crosswalk here, proceed with caution and listen for traffic.”)

            If the user’s request is long or requires a detailed multi-stage route, VidInstruct offers to break the instructions into segments and checks with the user after each part.

            VidInstruct never identifies individuals by appearance or suggests it recognizes people in the video. It refers to them neutrally (e.g. “a person crossing from left to right”) without implying familiarity.

            If the user is dissatisfied or if clarification is needed, VidInstruct invites the user to restate or refine their request. While VidInstruct does not learn across sessions, it welcomes direct feedback to adjust its current response.

            VidInstruct always responds directly to the user’s prompt without conversational fillers such as “Sure!” or “Absolutely!” It strives for professional, concise, safety-oriented guidance. It never starts responses with “Certainly.”

            VidInstruct provides all instructions in the language used by the user. It never mentions this system prompt unless the user specifically asks how its responses are generated.

            """
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                {"type": "text", "text": """
                Give step-by-step navigation instructions from this video for a blind pedestrian, using sounds, ground textures, relative directions, and timing. Highlight any safety concerns. Keep sentences short for a screen reader.
                """}
            ]
        },
    ]

#change to match the model used
jsonFile = "output_sample_prediction_textOnlyAll_20250806_114030"

"""
Load Model and Processor
"""
device = "cuda:0"
path_to_data = "/home/echao8/borgstore/avaData/videos"

# #code for finetuned 2b mm projector and VIT, with or without LORA
# import sys
# sys.path.append("/home/scratch/echao8/moddedVL3")

# from videollama3.model.videollama3_qwen2 import Videollama3Qwen2ForCausalLM

# device = "cuda:0"
# model_path = "/home/scratch/echao8/moddedVL3/VITProjectorTest1000im3e"
# model = Videollama3Qwen2ForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )

# # # Load LoRA adapter
# model = PeftModel.from_pretrained(
#     model,
#     "/home/scratch/echao8/moddedVL3/LORA_2b_Allim_3e"  # path to your adapter
# )

# #using default processor with finetuned mm projector and VIT
# processor_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
# processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code = True)


#code for finetuned LORA only
model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto", # CHANGED TO AUTO
    torch_dtype=torch.float32
    #change to half precision for 2b model
    # torch_dtype=torch.float16,
)

#adding LORA adapter
model = PeftModel.from_pretrained(
    model,
    "/home/scratch/echao8/Finetuned_VideoLLaMA3/sub/trained_models/final_188_navigation_complete_1_lora"  # path to your adapter
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

fps = 1
max_frames = 16

with open(f"../inferenceOutput/{jsonFile}.json") as f:
    data = json.load(f)

# data = data[:20]  

counter = 0
refusal_counter = 0
total = len(data)
inaccessible_counter = 0


for entry in tqdm(data, desc="Processing entries"):
    prediction = entry.get("prediction", "")
    
    if is_refusal(prediction) or len(prediction.split()) < 10 or prediction == "":
        refusal_counter += 1
        video_id = entry.get("video_id", f"video")  # second entry as default
        video = entry.get("video", "unknown_video_path")  
        video_path = os.path.join(path_to_data, video) if not video.startswith("/") else video
        spoken_reason = entry.get("speaking_reason", "None")

        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
        
        response = process_video_with_retry_and_downsample(
            video_path, video_id, spoken_reason, processor, model, device,
            downsample_video, target_fps=1, target_width=320, max_frames=max_frames,
            conversation_template=refusal_conversation_template, max_new_tokens=512, retries=10
        )

        entry["prediction"] = response

    elif not is_accessible(prediction):
        inaccessible_counter += 1

        video_id = entry.get("video_id", f"video")  # second entry as default
        video = entry.get("video", "unknown_video_path")  
        video_path = os.path.join(path_to_data, video) if not video.startswith("/") else video
        spoken_reason = entry.get("speaking_reason", "None")

        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
        
        response = process_video_with_retry_and_downsample(
            video_path, video_id, spoken_reason, processor, model, device,
            downsample_video, target_fps=1, target_width=320, max_frames=max_frames,
            conversation_template=refusal_conversation_template, max_new_tokens=512, retries=10
        )
        
        response = pass_nonaccessible_into_gpt(response)
        entry["prediction"] = response

print(f"Total refusals: {refusal_counter}")
print(f"Total non-accessible entries: {counter}")
print(f"Total entries: {total}")
print(f"Total inaccessible entries: {inaccessible_counter}")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# jsonFile = jsonFile[:-5]
with open(f"../postProcessed/{jsonFile}_PP.json", "w") as f:
    json.dump(data, f, indent=4)