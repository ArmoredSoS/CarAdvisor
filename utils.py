import json
import csv
from IPython.display import display, Markdown
import torch
from typing import Any, Callable, Dict, Tuple
from functools import partial
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from models import qwen3

device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

MODELS: Dict[str, Tuple[str, Callable[..., Any], Callable[..., Any]]] = {
    "qwen3": (
        "Qwen/Qwen3-4B-Instruct-2507",
        partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True, quantization_config=quantization_config, device_map=device_map),
        qwen3.prepare_text,
    )
}

def display_conversation(messages, user_msg, assistant_msg):
    md = (
        "### Conversation\n\n"
        f"**System:** {messages[0]['content']}\n\n"
        f"**User:** {user_msg}\n\n"
        f"**Assistant:** {assistant_msg}"
    )
    display(Markdown(md))

class CSVPreprocessing:
    
    def __init__(self, file_path):
        self.file_path = file_path

    def get_slots(self):
        columns = []
        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            columns = next(reader)
        return columns
    
    def get_possible_values(self, slot):
        return {row[slot] for row in self.get_cars() if row[slot]}
    
    def get_cars(self):
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            return [row for row in reader]
        
    def get_all_possible_values(self):
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)

            result = {field: set() for field in reader.fieldnames}

            for row in reader:
                for field in reader.fieldnames:
                    value = row[field]
                    if value:
                        result[field].add(value)

        return result
    
DATASET = CSVPreprocessing("vehicles.csv")
SLOTS = DATASET.get_slots()
CARS = DATASET.get_cars()
possible_values = DATASET.get_all_possible_values()
model = MODELS["qwen3"]

class CarRecommender:
    """The CarRecommender class is responsible for scoring and ranking cars based on user preferences extracted from the conversation."""

    def __init__(self, top_k=3):
        self.top_k = top_k

    @staticmethod
    def score_car(car, slots):

        score = 0

        for slot, value in slots.items():
            if value == "":
                continue

            car_value = car.get(slot)
            if car_value is None:
                continue

            car_str = str(car_value).lower().strip()
            value_str = str(value).lower().strip()

            if value_str in car_str or car_str in value_str:
                score += 1

        return score

    @staticmethod
    def rank(cars, slots):
        return sorted(
            [(car, CarRecommender.score_car(car, slots)) for car in cars],
            key=lambda x: x[1],
            reverse=True
        )
        
    def recommend(self, slots):
        ranked = self.rank(CARS, slots)
        filtered = [(car, score) for car, score in ranked if score > 0]
        return filtered[:self.top_k]

class NLU:

    def __init__(self, model, tokenizer, prepare_text, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_text = prepare_text
        self.system_prompt = system_prompt

    def parse(self, user_msg, messages, n_exchanges):
        history = messages[:-1][-n_exchanges:]

        transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)

        full_user_msg = (
            f"Conversation so far:\n{transcript}\n\n"
            f"Latest user message to analyze:\n{user_msg}\n\n"
            "Respond ONLY with the JSON object in the format specified in the system "
            "prompt. Do not continue the conversation, do not add explanations or any "
            "other text."
        )

        nlu_messages = [{"role": "system", "content": self.system_prompt}]

        text = self.prepare_text(full_user_msg, self.tokenizer, nlu_messages, 0)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256).cpu()

        decoded = self.tokenizer.decode(
            output[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        # Strip Qwen3 thinking blocks, gave issues with JSON parsing and are not relevant for NLU output
        import re
        decoded = re.sub(r'<think>.*?</think>', '', decoded, flags=re.DOTALL).strip()

        start = decoded.find('{')
        end = decoded.rfind('}') + 1

        if start == -1 or end <= start:
            raise ValueError(f"No JSON object found in NLU output: {repr(decoded)}")

        json_str = decoded[start:end]
        result = json.loads(json_str)
        
        return result
    
class DM:

    def __init__(self, model, tokenizer, prepare_text, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_text = prepare_text
        self.system_prompt = system_prompt

    def validate(self, slots):
        errors = {}
        for slot, value in slots.items():
            if value == "" or slot not in possible_values:
                continue
            value_str = str(value).lower().strip()
            valid_strs = {str(v).lower().strip() for v in possible_values[slot]}
            if not any(value_str in v or v in value_str for v in valid_strs):
                errors[slot] = "Invalid"
        return errors

    def decide(self, state):
        errors = self.validate(state["slots"])
        state = {**state, "errors": errors}

        content = f"\nDialogue state:\n{json.dumps(state)}\n"
        dm_messages = [{"role": "system", "content": self.system_prompt + content}]

        text = self.prepare_text("", self.tokenizer, dm_messages, 0)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=128).cpu()

        decoded = self.tokenizer.decode(
            output[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )

        import re
        decoded = re.sub(r'<think>.*?</think>', '', decoded, flags=re.DOTALL).strip()

        start = decoded.find('{')
        end = decoded.rfind('}') + 1
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object found in DM output: {repr(decoded)}")

        json_str = decoded[start:end]

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            action_match = re.search(r'"action"\s*:\s*"([^"]+)"', json_str)
            if action_match:
                return action_match.group(1), None
            return "slot_filling", None

        return result.get("action"), result.get("value")

class NLG:
    def __init__(self, model, tokenizer, prepare_text, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_text = prepare_text
        self.system_prompt = system_prompt

    def generate(self, nba, ds, extra=None, messages=None, n_exchanges=10):
        history = (messages or [])[-n_exchanges:]
        transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)

        content = (
            f"Conversation so far:\n{transcript}\n\n"
            f"Next best action: {nba}\n"
            f"Dialogue state: {json.dumps(ds)}\n\n"
            f"Additional info (this is the ONLY car data you may reference — do not "
            f"invent any vehicle, model, manufacturer, or specification not listed here):\n"
            f"{extra if extra else 'None'}\n\n"
            "Now write the next assistant message."
        )

        nlg_messages = [{"role": "system", "content": self.system_prompt}]
        text = self.prepare_text(content, self.tokenizer, nlg_messages, 0)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=300).cpu()

        decoded = self.tokenizer.decode(
            output[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        return decoded.strip()