from PIL import Image
import os
import torch
import json
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "omkarthawakar/LlamaV-o1"


model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()
processor = AutoProcessor.from_pretrained(model_id)


def generate_inner(question, image):
    kwargs = {
        'max_new_tokens': 2048,
        "top_p": 0.9,
        "pad_token_id": 128004,
        "bos_token_id": 128000,
        "do_sample": True,
        "eos_token_id": [
            128001,
            128008,
            128009
        ],
        "temperature": 0.6,
        "num_beams": 8,
        "use_cache": True,

    }
    messages = [
        {
            'role': 'user', 
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': question+"\nPlease generate a summary of the picture."}
            ],
        }
    ]

    def infer(messages: dict) -> str:
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors='pt').to(model.device)
        output = model.generate(**inputs, **kwargs)
        return processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace("<|end_of_text|>", "")

    def tmp(inp, out):
        return [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': inp}
                ]
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': out}
                ]
            }
        ]
    out = infer(messages)
    caption_prompt = "Please generate a detailed caption for the image"
    messages.extend(tmp(out, caption_prompt))
    out = infer(messages)
    reasoning_prompt = "Please generate a detailed reasoning to answer the question given the caption."
    messages.extend(tmp(out, reasoning_prompt))
    reasoning = infer(messages)
    conclusion_prompt = "Please generate the final answer. Do not output anything else."
    messages.extend(tmp(reasoning, conclusion_prompt))
    out = infer(messages)
    print(f"Question: {question}\nAnswer: {out}")
    return out, reasoning    


def reasoning_steps_answer(img, question, choices):
    
    predicted_answer, reasoning = generate_inner(question, img)
    return predicted_answer, reasoning

all_data = []
json_paths = "path/to/json/files"
image_path = "path/to/images"
start = 500
end = 1500
for file in tqdm(os.listdir(json_paths)):
    if not file.endswith(".json"): continue
    with open(f"{json_paths}/{file}", "r") as json_file:
        data = json.load(json_file)
        try:
            image = Image.open(f"{image_path}/{data['image']}")
            question = data["question"]
            final_answer = data["final_answer"]
            idx = data["idx"]
            reasoning_answer = data["answer"]
            question += "\nPlease select the correct option by its letter." if "Choices" in question else ""
            model_answer, reasoning = generate_inner(question, image)
            
            all_data.append({
                "idx": idx,
                "question": question,
                "final_answer": final_answer,
                "answer": reasoning_answer,
                "llm_response": reasoning+"\n\n\n"+model_answer,
            })
        except Exception as e:
            print("Skipping file", file, "for", e)
            continue

model_pref = model_id.replace("/", "_")
with open(f"results_llamaV-o1.json", "w") as json_file:
    json.dump(all_data, json_file, indent=4)
