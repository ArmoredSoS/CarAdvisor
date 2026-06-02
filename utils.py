import json

from IPython.display import display, Markdown
import torch
from typing import Any, Callable, Dict, Tuple
from functools import partial

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from models import qwen3

#NOTICE: AI generated code has been used to increase the efficiency of the model and prevent excessive memory usage

# Quantization config for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Determine device map
device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

# You can add more models here as needed
# The tuple contains the model name, a partial function with the model specific arguments, the method to prepare the input text
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


class CarRecommender:

    """The CarRecommender class is responsible for scoring and ranking cars based on user preferences (slots),
    using a priority system to weigh different attributes according to their importance in the recommendation process.
    """

    PRIORITY = {
        "car_price": 4,
        "car_type": 3,
        "car_usecase": 4,
        "fuel_efficiency": 3,
        "car_brand": 1,
        "car_dimensions": 1,
        "fuel_type": 2,
        "car_design": 1
    }

    @staticmethod
    def score_car(car, slots):
        
        """Score a single car based on how well it matches the user's preferences (slots)."""
        
        score = 0

        for slot, value in slots.items():
            # Skip null or None values
            if value is None or value == "null" or value == "":
                continue

            car_value = car.get(slot)
            if car_value is None:
                continue

            if slot == "car_price":
                try:
                    car_price = int(car_value)
                    user_budget = int(value) if isinstance(value, int) else int(float(value))

                    if car_price <= user_budget:
                        score += CarRecommender.PRIORITY[slot]
                    else:
                        score -= 2
                except (TypeError, ValueError):
                    pass

            else:
                # String comparison - check for exact match or substring match
                car_str = str(car_value).lower().strip()
                value_str = str(value).lower().strip()
                
                if car_str == value_str or value_str in car_str or car_str in value_str:
                    score += CarRecommender.PRIORITY.get(slot, 1)

        return score

    @staticmethod
    def rank(cars, slots):
        
        """Score and rank the cars based on the provided slots. 
        The scoring is done by comparing each car's attributes with the user's preferences 
        (slots) and applying a priority system to weigh different attributes according to 
        their importance in the recommendation process.
        """
        
        scored = [(car, CarRecommender.score_car(car, slots)) for car in cars]
        scored.sort(key=lambda x: x[1], reverse=True)
        # Debug: print top 5 scored cars
        #print(f"\n[DEBUG] Slots for filtering: {slots}")
        #print(f"[DEBUG] Top 5 scored cars:")
        #for car, score in scored[:5]:
        #    print(f"  {car.get('car_brand', 'Unknown')}: score={score}")
        return scored
    
class RecommenderService:

    """The RecommenderService class is responsible for managing the car recommendation process."""

    def __init__(self, cars, engine=None):
        self.cars = cars
        self.engine = engine or CarRecommender

    def car_exists(self, car_name):
        """Check if a car with the given name exists in the dataset (case-insensitive)"""
        return any(
            car_name.lower().strip() == car["car_brand"].lower().strip()
            for car in self.cars
        )

    def recommend(self, slots, top_k=3):
        """Recommend cars based on the provided slots, returning only those that exist in the dataset"""
        ranked = self.engine.rank(self.cars, slots)

        # Keep only cars that actually exist in dataset
        validated = [
            (car, score)
            for car, score in ranked
            if self.car_exists(car["car_brand"])
        ]

        return validated[:top_k]

    def format_recommendation(self, ranked):
        """Format the ranked list of cars into a user-friendly string"""
        return [car["car_brand"] for car, _ in ranked]

    def get_car_by_brand(self, brand):
        """Strict match for exact brand names"""
        for car in self.cars:
            if car.get("car_brand", "").lower().strip() == brand.lower().strip():
                return car
        return None
    
    def get_cars_by_brand(self, brands):
        """Get multiple cars by their brand names"""
        return [self.get_car_by_brand(brand) for brand in brands if self.get_car_by_brand(brand) is not None]
    
    def compare_cars(self, brands):
        """Compare multiple cars by their brand names"""
        cars = self.get_cars_by_brand(brands)
        
        if not cars or len(cars) < 2:
            return None
        
        comparison = {}
        for car in cars:
            comparison[car["car_brand"]] = {
                "car_price": car.get("car_price"),
                "car_type": car.get("car_type"),
                "fuel_type": car.get("fuel_type"),
                "car_usecase": car.get("car_usecase"),
                "car_state": car.get("car_state"),
                "car_dimensions": car.get("car_dimensions"),
                "fuel_efficiency": car.get("fuel_efficiency"),
                "car_design": car.get("car_design")
            }
            
        return comparison


class NLU:
    
    """The NLU component is responsible for parsing user messages and extracting 
    structured information such as intent and slots. It uses a language model to 
    perform this parsing based on the conversation history and a system prompt 
    that defines the expected output format."""
    
    def __init__(self, model, tokenizer, prepare_text, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_text = prepare_text
        self.system_prompt = system_prompt

    def parse(self, user_msg, messages, n_exchanges):
        nlu_messages = (
            [{"role": "system", "content": self.system_prompt}]
            + [m for m in messages[-n_exchanges:] if m["role"] == "user"]
        )
        text = self.prepare_text(user_msg, self.tokenizer, nlu_messages, 0)
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
    
class DialogueManager:
    def __init__(self, rules):
        self.rules = rules
        self.slots = [    
            "car_brand",
            "car_price",
            "car_type",
            "fuel_type",
            "car_usecase",
            "car_state",
            "car_dimensions",
            "fuel_efficiency",
            "car_design"
        ]
        # Core slots are the essential slots for car search
        self.core_slots = [
            "car_type",
            "car_price",
            "car_state",
            "car_usecase",
            "fuel_efficiency"
        ]
        
        self.optional_slots = [
            slot for slot in self.slots
            if slot not in self.core_slots
        ]

    def validate(self, slots):
        validated = {}
        errors = {}

        for slot, value in slots.items():

            # normalize text
            if isinstance(value, str):
                value = value.lower().strip()

            if value in ("null", None):
                validated[slot] = None
                continue

            if isinstance(value, list) and slot == "comparison_cars":

                valid_cars = []
                invalid_cars = []

                for car in value:

                    # normalize each car
                    if isinstance(car, str):
                        car = car.lower().strip()

                    # validate against car_brand rule
                    if self.rules.get("car_brand") and self.rules["car_brand"](car):
                        valid_cars.append(car)
                    else:
                        invalid_cars.append(car)

                # At least 2 valid cars required for comparison
                if len(valid_cars) >= 2:
                    validated[slot] = valid_cars

                    # optionally track invalid entries too
                    if invalid_cars:
                        errors[slot] = {
                            "invalid_cars": invalid_cars
                        }

                else:
                    validated[slot] = None

                    errors[slot] = {
                        "invalid_cars": invalid_cars
                    }

                continue

            # Normalize price strings
            if slot == "car_price" and isinstance(value, str):
                try:
                    normalized = int(value.replace(",", ""))
                    value = normalized
                except ValueError:
                    # Allow numeric strings with k/K notation
                    if value.endswith("k") and value[:-1].replace(".", "", 1).isdigit():
                        try:
                            value = int(float(value[:-1]) * 1000)
                        except ValueError:
                            pass

            rule = self.rules.get(slot)

            if rule and rule(value):
                validated[slot] = value
            else:
                validated[slot] = None
                errors[slot] = value

        return validated, errors

    def decide(self, state):
        # Handle special intents immediately without validation
        if state["intent"] == "ignore_message":
            return "ignore", None
        
        if state["intent"] == "car_info":
            return "provide_car_description", None

        slots, errors = self.validate(state["slots"])

        if state["intent"] == "car_search" and errors:
            return "slot_filling_error", str(list(errors.keys())[0])
        
        if state["intent"] == "car_comparison":
            if errors:
                
                return "slot_filling_error", errors["comparison_cars"]["invalid_cars"]
                #return "slot_filling_error", "comparison_cars"
            else:
                return "compare_cars", None
            
        if state["intent"] == "end_conversation":
            return "end_conversation", None
            
        filled_core = [s for s in self.core_slots if slots.get(s) not in ("null", None, "")]
        filled_optional = [s for s in self.optional_slots if slots.get(s) not in ("null", None, "")]

        if len(filled_core) >= 3:
            return "recommend", None
        elif len(filled_core) >= 2 and len(filled_optional) >= 2:
            return "recommend", None
        else:
            missing = [s for s in self.core_slots if slots.get(s) in ("null", None, "")]
            if missing:
                return "slot_filling", f"{missing[0]}"
            else:
                return "recommend", None
        
        

class NLG:
    def __init__(self, model, tokenizer, prepare_text, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_text = prepare_text
        self.system_prompt = system_prompt

    def generate(self, nba, ds, extra=None, messages=None, n_exchanges=10):
        #print(f"\n[DEBUG] Generating response with NBA: {nba} and DS: {ds}")
        content = f"""
            NBA: {nba}
            DS: {json.dumps(ds)}

            Additional info:
            {extra if extra else ""}
            """

        messages = ([{"role": "system", "content": self.system_prompt + content}]+ messages[-n_exchanges:])
        #messages = [{"role": "system", "content": self.system_prompt + content}]
        
        text = self.prepare_text("", self.tokenizer, messages, n_exchanges)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=260).cpu()

        decoded = self.tokenizer.decode(
            output[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        return decoded.strip()
   