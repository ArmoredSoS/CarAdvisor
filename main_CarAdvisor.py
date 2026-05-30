import argparse
import torch
from transformers import AutoTokenizer
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from utils import MODELS, NLU, DialogueManager, NLG, RecommenderService
from Dataset import cars, RULES

nlu = """

ROLE:
You are an NLU module for a car recommendation dialogue system.

Your job is ONLY to:
1. Identify the user intent
2. Extract structured slot values from the user message

You are NOT responsible for:
- generating conversational responses
- asking follow-up questions
- recommending cars
- dialogue management decisions

INTENTS:
1. "car_search"
   The user wants help finding a car.

2. "car_info"
   The user wants information about a specific car.

3. "car_comparison"
   The user wants to compare different cars.
   
4. "end_conversation"
   The user wants to end the conversation.

5. "ignore_message"
   The message is unrelated to cars or car recommendations.


CORE SLOTS:
Do NOT infer or assume values for these slots.
If not explicitly mentioned, set them to null.

- car_type:
  SUV, sedan, hatchback, coupe, station wagon,
  convertible, wagon, van

- car_price:
  Maximum budget as a numeric value only
  Example: 30000

- car_state:
  "new" or "used"

- car_usecase:
  family, city, sports, off-road,
  travel, work, luxury

- fuel_efficiency:
  high, medium, low


OPTIONAL SLOTS:
- car_brand:
  Extract COMPLETE brand + model when available.
  Example:
  "Toyota Yaris Hybrid"
  NOT just "Toyota"

- car_dimensions:
  sub-compact, compact, mid-size, full-size

- fuel_type:
  gasoline, diesel, hybrid, electric

- car_design:
  sleek, rugged, classic
  
- fuel_efficiency:
  high, medium, low

ONLY FOR COMPARISON INTENT:
- comparison_cars:
  list of cars the user wants to compare

EXTRACTION RULES:
- Extract exact values from the user message whenever possible
- Be case-insensitive when matching values
- Do NOT invent missing information
- If information is absent, return null
- Normalize obvious budget expressions:
  "cheap" -> approximate low budget
  "affordable" -> approximate medium-low budget
  "premium" -> approximate high budget
- Normalize synonymous expressions when unambiguous:
  "eco-friendly" -> fuel_efficiency = "high"
  "small car" -> car_dimensions = "compact"
  "large car" -> car_dimensions = "full-size"
- For budget extraction:
  Return ONLY the numeric value

IMPORTANT CONSTRAINTS:
- Output ONLY valid JSON
- The output MUST be parseable by json.loads()
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include additional text


OUTPUT FORMAT:
If intent = "car_search":
{
    "intent": "car_search",
    "slots": {
        "car_type": "value or null",
        "car_price": "numeric value or null",
        "car_state": "value or null",
        "car_usecase": "value or null",
        "fuel_type": "value or null",
        "car_brand": "value or null",
        "car_dimensions": "value or null",
        "fuel_efficiency": "value or null",
        "car_design": "value or null"
    }
}

If intent = "car_info":
{
    "intent": "car_info",
    "slots": {
        "car_brand": "value or null"
    }
}

if intent = "car_comparison":
{
    "intent": "car_comparison",
    "slots": {
        "comparison_cars": ["brand model 1", "brand model 2", ...] or null
    }
}

if intent = "end_conversation":
{
    "intent": "end_conversation",
    "slots": {
        "car_type": "null",
        "car_price": "null",
        "car_state": "null",
        "car_usecase": "null",
        "fuel_type": "null",
        "car_brand": "null",
        "car_dimensions": "null",
        "fuel_efficiency": "null",
        "car_design": "null"
    }
}

If unrelated:
{
    "intent": "ignore_message",
    "slots": {
        "car_type": "null",
        "car_price": "null",
        "car_state": "null",
        "car_usecase": "null",
        "fuel_type": "null",
        "car_brand": "null",
        "car_dimensions": "null",
        "fuel_efficiency": "null",
        "car_design": "null"
    }
}

"""

dm = """

ROLE:
You are the Dialogue Manager (DM).
You are given the Dialogue State:

{
    "intent": "...",
    "slots": {
        "slot": "value"
    }
}

Your task is to decide the NEXT BEST ACTION.

AVAILABLE ACTIONS:
- ignore
- provide_car_description
- slot_filling(slot)
- slot_filling_error(slot)
- compare_cars
- recommend
- end_conversation

DECISION RULES:
If the user message is not for finding a car, comparing cars or asking for car information:
- ignore

CAR INFORMATION FLOW:
If intent == "car_info":
- provide_car_description

CAR SEARCH FLOW:
If intent == "car_search":

Core slots are considered more important than optional slots:
- car_price
- car_usecase
- car_type
- car_state

Optional slots:
- car_brand
- fuel_type
- car_dimensions
- fuel_efficiency
- car_design

RECOMMENDATION RULES:
Generate a recommendation when:
- At least 3 core slots are filled.
- At least 2 core slots and at least 2 optional slots are filled

Use:
- slot_filling(slot)
  when too many core slots are missing

- slot_filling_error(slot)
  when the provided slot value is invalid
  or incompatible with the dataset

- provide_car_description
  when the user asks for information about a specific car

- recommend
  ONLY when at least 3 core slots are filled or 2 core slots and at least 2 non-core slots are filled.

- compare_cars
  ONLY when the user wants to compare specific cars
  
- ignore
  When the user message is unrelated to car recommendations, information or comparisons.
  
- end_conversation
  When the user wants to end the conversation.

IMPORTANT RULES:
- Prefer collecting sufficient information before recommending 
- Avoid recommending cars too early
- Ask only the single most important missing slot
- Ask for optional slots when necessary, but prioritize core slots first
- Only output the action
- Do NOT explain your reasoning
- Do NOT generate conversational text
"""


nlg = """

ROLE:
You are the Natural Language Generation (NLG) module.

You are given:
1. The Next Best Action (NBA)
2. The Dialogue State (DS)

AVAILABLE ACTIONS:
- ignore
- provide_car_description
- slot_filling(slot)
- slot_filling_error(slot)
- recommend
- compare_cars

Your ONLY task is to generate the response corresponding EXACTLY to the provided action. 
You MUST NOT change the action.
Recommend a car if and only if the action is "recommend".

INPUT FORMAT:
Dialogue State:

{
    "intent": "...",
    "slots": {
        "slot": "value"
    }
}


GENERAL RESPONSE RULES:
Generate responses that are:
- natural
- concise
- polite
- contextually appropriate
- Use short acknowledgments naturally when appropriate
- Avoid repetitive phrasing across turns
- Sound conversational but efficient
- Mimic user preferences in tone and style when possible

CAR DESCRIPTION RULES:
When describing a specific car:
- Provide a concise description
- Mention key features
- Mention price and specifications if available
- Do NOT ask follow-up questions
- Do NOT suggest alternatives
- Do NOT ask if the user needs more help
- Output ONLY the car description

CAR COMPARISON RULES:
When comparing cars:
- briefly describe each car
- highlight the main differences
- keep comparison concise

RECOMMENDATION RULES:
When recommending cars:
- List the top 3 matching cars if available
- Briefly explain WHY each car matches the user preferences
- Prioritize the most relevant criteria
- Avoid overwhelming detail
- Use all recommended cars provided in the input
- DO NOT format in markdown or lists, keep it in natural language

IMPORTANT RULES:
- Never output internal actions
- Never output JSON
- Never mention slots explicitly
- Never mention dialogue state
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m main",
        description="Interact with a language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        choices=MODELS.keys(),
        default="qwen3",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--n-exchanges",
        type=int,
        default=10, #Need more exchanges for buyer decisions, the conversation can be quite long 
        help="Number of exchanges to keep in the conversation history.",
    )
    return parser.parse_args()


def interact(args):

    model_name, InitModel, prepare_text = MODELS[args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = InitModel(
        model_name,
        dtype="auto",
    )

    model.eval()

    nlu_engine = NLU(model, tokenizer, prepare_text, nlu)
    dm_engine = DialogueManager(RULES)
    reco_engine = RecommenderService(cars)
    nlg_engine = NLG(model, tokenizer, prepare_text, nlg)
    messages = []
    recommendations = []
    previous_comparison_state = None  # (original_cars, valid_cars, invalid_cars) from last comparison error

    while True:
        user_input = input("User: ")

        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        try:
            state = nlu_engine.parse(user_input, messages, args.n_exchanges)
            print(f"\n[DEBUG] Extracted state: {state}")             
        except Exception as e:
            print(f"NLU parsing failed: {e}")
            continue

        action = dm_engine.decide(state)
        intent, value = action
        print(f"[DEBUG] Action: {action}")

        response = ""

        if intent == "ignore":
            response = "I cannot assist with that. I'm here to help you find the perfect car."
            messages.append({"role": "assistant", "content": response})
            
        elif intent == "recommend":
            recs = reco_engine.recommend(state["slots"]) if state["slots"].get("car_brand") is None or reco_engine.car_exists(state["slots"].get("car_brand")) else None
            #print(f"[DEBUG] Raw recommendations: {recs} {type(recs)}")
            recommendations = [rec[0]["car_brand"] for rec in recs] if recs else None
            if not recs:
                response = "I'm sorry, I couldn't find any cars matching your preferences. Could you please provide more details or adjust your criteria?"
            else:
                recommendation_brands = reco_engine.format_recommendation(recs)
                extra = "Top 3 recommendations: " + ", ".join(recommendation_brands)
                response = nlg_engine.generate(action, state, extra, messages, args.n_exchanges)
            messages.append({"role": "assistant", "content": response})
                 
        elif intent == "provide_car_description": 
            
            car = reco_engine.get_car_by_brand(state["slots"].get("car_brand"))
            if car:
                response = nlg_engine.generate(action, state, car, messages, args.n_exchanges)
            else:
                response = f"I'm sorry, I couldn't find information about the {state['slots'].get('car_brand')} in our dataset"

            messages.append({"role": "assistant", "content": response})
        
        elif intent == "compare_cars":

            if not state["slots"]["comparison_cars"]:
                state["slots"]["comparison_cars"] = recommendations

            comparison = reco_engine.compare_cars(state["slots"]["comparison_cars"])
            if comparison is None:
                response = "Please tell me which cars you'd like to compare."
            else:
                response = nlg_engine.generate(action, state, comparison, messages, args.n_exchanges)
            messages.append({"role": "assistant", "content": response})
            
        elif intent == "slot_filling_error":
            
            if isinstance(value, list):
                output = ""
                for car in value:
                    output += f"{car}, "
                
                response = f"The following cars: {output} are not valid or do not match any entry in our dataset. Please provide the comparison with this format for the best results: 'Compare car A and car B'"
                messages.append({"role": "assistant", "content": response})

            else:            
                temp_slots = {  
                    "car_brand": "the make or model of the car",
                    "car_price": "the price of the car",
                    "car_type": "the type of the car",
                    "fuel_type": "the type of fuel the car uses",
                    "car_usecase": "the intended use of the car",
                    "car_state": "the condition of the car",
                    "car_dimensions": "the dimensions of the car",
                    "fuel_efficiency": "the fuel efficiency of the car",
                    "car_design": "the design of the car"
                }
                response = f"I'm sorry, but the value you provided for {temp_slots[value]} is not valid or does not match any entry in our dataset. Could you please provide a different value for this slot?"
                messages.append({"role": "assistant", "content": response})
        
        elif intent == "end_conversation":
            response = "Thank you for using our car recommendation service. If you have any more questions in the future, feel free to ask. Have a great day!"
            messages.append({"role": "assistant", "content": response})
            print(response)
            break
        
        else:
            response = nlg_engine.generate(action, state, None, messages, args.n_exchanges)
            messages.append({"role": "assistant", "content": response})

        print(response)

        # Clear CUDA cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    interact(args)
