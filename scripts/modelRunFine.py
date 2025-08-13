import os
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoImageProcessor
from tqdm import tqdm
import ffmpeg
from peft import PeftModel


def downsample_video(input_path, output_path, target_fps=1, target_width=320, duration=None):
    stream = ffmpeg.input(input_path)
    if duration is not None:
        stream = stream.trim(start=0, duration=duration).setpts('PTS-STARTPTS')
    stream = stream.filter('fps', fps=target_fps)
    stream = stream.filter('scale', target_width, -1)
    stream = ffmpeg.output(stream, output_path)
    stream = stream.overwrite_output()
    stream.run(quiet=True)

"""
Load Model and Processor
"""
#code for finetuned 2B mm projector and VIT, with or without LORA
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
device = "cuda:0"

model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto", # CHANGED TO AUTO
    #change to half precision for 2B model
    torch_dtype=torch.float32,
    # torch_dtype = torch.float16
)

# # Load LoRA adapter
model = PeftModel.from_pretrained(
    model,
    # "/home/scratch/echao8/moddedVL3/LORA_2b_1000im_3e"  # path to your adapter,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code = True)


"""
Load data
"""
path_to_data = "/home/echao8/borgstore/avaData/videos"
# path_to_data = "../testData" 
file_type = "sample_prediction"
# file_type = "single_video"
# file_type = "few"
model_type = "2b_baseline"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"../inferenceOutput/output_{file_type}_{model_type}_{timestamp}.json"
file_type = "../dataAnnotations/" + file_type + ".json"  


"""
Process each entry in the json
"""
all_responses = []
with open(file_type, "r") as f:
    data = json.load(f)


"""
Video processing parameters
"""
fps = 1
max_frames = 16

sys_prompt = """
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

usr_prompt = """
Give step-by-step navigation instructions from this video for a blind pedestrian, using sounds, ground textures, relative directions, and timing. Highlight any safety concerns. Keep sentences short for a screen reader.
"""
downsample_width = 320

print(sys_prompt)
print(usr_prompt)
print(fps, max_frames)
print(f"Downsampling width: {downsample_width}")


for i, conversation in enumerate(tqdm(data, desc="Processing videos")):
    video_id = conversation.get("video_id", f"video_{i}")  # second entry as default
    video = conversation.get("video", "unknown_video_path")  
    video_path = os.path.join(path_to_data, video) if not video.startswith("/") else video
    spoken_reason = conversation.get("speaking_reason", "None")

    downsampled_path = f"/tmp/downsampled_{video_id}.mp4"
    downsample_video(video_path, downsampled_path, target_fps=1, target_width=downsample_width, duration=max_frames)
    video_path = downsampled_path
    
    conversation = [
    {
    "role": "system",
    "content": sys_prompt
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
            {"type": "text", "text": usr_prompt
            ,}
        ]
    },
    ]


    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ) 
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        #change to half precision for 2B model
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        # inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    output_ids = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


    all_responses.append({
        "id": i,
        "video_id": video_id,
        "video": video,
        "prediction": response,
        "spoken_reason": spoken_reason
    })

    #clear gpu mem
    os.remove(downsampled_path)

    del inputs, output_ids, response
    torch.cuda.empty_cache()

with open(output_file, "w") as f:
    json.dump(all_responses, f, indent=4)

print(f"Results saved to {output_file}")

print("Done.")
