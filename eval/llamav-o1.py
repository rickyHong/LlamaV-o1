import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "omkarthawakar/LlamaV-o1"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("../MathVista_74.png")


def generate_inner(question, image):
    kwargs = {
        'max_new_tokens': 2048,
        "top_p": 0.9,
        "pad_token_id": 128004,
        "bos_token_id": 128000,
        "do_sample": False,
        "eos_token_id": [
            128001,
            128008,
            128009
        ],
        "temperature": 0.1,
        "num_beams": 1

    }
    messages = [
        {
            'role': 'user', 
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': question+"\nSummarize how you will approach the problem and explain the steps you will take to reach the answer."}
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
    caption_prompt = "Provide a detailed description of the image, particularly emphasizing the aspects related to the question."
    messages.extend(tmp(out, caption_prompt))
    out = infer(messages)
    reasoning_prompt = "Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning."
    messages.extend(tmp(out, reasoning_prompt))
    reasoning = infer(messages)
    conclusion_prompt = "State the final answer in a clear and direct format. It must match the correct answer exactly."
    messages.extend(tmp(reasoning, conclusion_prompt))
    out = infer(messages)
    return out, reasoning    


print(generate_inner("How many objects are preferred by more than 90 percent of people in at least one category?", image))
