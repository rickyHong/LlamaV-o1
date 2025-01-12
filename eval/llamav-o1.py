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

url = "https://datasets-server.huggingface.co/assets/VIM-Bench/VIM-MathVista/--/d6ba926dc6b088c3fc7241f1d0155aa1ec9fbe2f/--/default/vim_mathvista_testmini/12/decoded_image/image.png?Expires=1734560868&Signature=bdSm11mQAW3GMBxY7lEv5cobNeRd6lV4bW42ueD6VN~Hi--s-NhmnA3D7GT6vfhDvYsnDykHKFA07EI7JHgYo-jB1O9BI2xEicoxpACShVfrklV9XtQZaNVEUi0pBwOgWSJmj4OmjeqZKRe6-NBPvWRDcB0HJ0jmN1-~CaRO2BACJqlqjzckJ6mMkSYqnyHNp3OFfXWN-uRrTWLw6FttonQyj~~dqKy1bA9CxDQuL3lE3DuSf1dqc7O1ZqgiRjyA~rRiJc-3XqlaBwbDqU9zj9~WU02WN0epaty3lq4vX-zQwR3Qd2pXj5WuY3XiBcRiP60iNFUAijjOBeQCyJ7g0w__&Key-Pair-Id=K3EI6M078Z3AC3"
image = Image.open(requests.get(url, stream=True).raw)


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
        "num_beams": 8

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
    return out, reasoning    


print(generate_inner("How many objects are preferred by more than 90 percent of people in at least one category?", image))
