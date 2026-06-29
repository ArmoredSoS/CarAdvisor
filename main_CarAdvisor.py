import argparse
import torch
from transformers import AutoTokenizer
import os
import warnings
from utils import NLU, DM, NLG, CarRecommender, MODELS, SLOTS
import json

warnings.filterwarnings("ignore")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

nlu = """
You are an NLU module for a car recommendation dialogue system.

ROLE:
- Identify the user intent
- Extract structured slot values from the user message

INTENTS:
- provide_information: the user provides information about themselves or the car they are looking for.
- require_information: the user needs help from the system or asks questions.
- correct_information: the user corrects previously provided information.
- request_recommendation: the user asks for a car recommendation.
- end_conversation: the user explicitly wants to end the conversation.

AVAILABLE SLOTS WITH FORMATTING RULES AND DESCRIPTION:
- model_year: year of the car, expressed as a 4-digit number (only 2026 cars available)
- manufacturer: the company that produces the car (e.g., Toyota, Ford)
- model: the specific model of the car (e.g., Camry, Mustang)
- vehicle_class: the class/style of the car (Two Seaters, Minicompact Cars, Subcompact Cars,Compact Cars,
                            Midsize Cars, Large Cars, Small Station Wagons, Midsize Station Wagons,
                            Small Pick-up Trucks 2WD, Small Pick-up Trucks 4WD, Standard Pick-up Trucks 2WD,
                            Standard Pick-up Trucks 4WD, Small SUV 2WD,Small SUV 4WD, Standard SUV 2WD','Standard SUV 4WD)
- fuel_type: the type of fuel the car uses (gasoline, diesel, electricity)
- engine_displacement_l: engine displacement expressed in liters (e.g., 2.0, 3.5)
- cylinders: the number of cylinders in the engine of a combustion car (e.g., 4, 6, 8)
- transmission: the type of transmission (manual, auto)
- drivetrain: type of drivetrain and traction system ("2-Wheel Drive, Front" or "2-Wheel Drive, Rear" or "4-Wheel Drive" or "All-Wheel Drive")
- cargo_volume_cu_ft: cargo volume expressed in cubic feet (range 5-35 cu ft)
- combined_mpg: combined fuel efficiency expressed in miles per gallon (range 10-150 mpg)
- driving_range_miles: driving range expressed in miles (range 100-500 miles)
- electric_range_miles: electric range expressed in miles (range 5-50 miles)
- phev_blended_mpge: plug-in hybrid blended fuel efficiency expressed in MPGe (range 10-110 mpge)
- annual_fuel_cost_usd: annual fuel cost expressed in US dollars (range 500-5000 usd)
- co2_emissions_g_per_mile: CO2 emissions expressed in grams per mile (range 0-600 gpm)
- fe_rating_1_10: fuel efficiency rating expressed on a scale from 1 to 10

EXTRACTION RULES:
- Extract exact values following the formatting rules and descriptions from the user message whenever possible, be case-insensitive and ignore irrelevant words
    - If the user changes values for a slot that is already filled, update it with the new information
- Infer realistic values respecting the slot list the, formatting rules and descriptions provided above
    - If the inferred values are for already filled slots, update them with the new information
- If information is absent, leave the slot empty

OUTPUT FORMAT:
If intent = "provide_information" or "request_recommendation":
{
    "intent": "provide_information" or "request_recommendation",
    "slots": {
        "slot_name": value or empty string
        ...
    }
}

If intent = "require_information":
{
    "intent": "require_information",
    "slots": {
        "required_slots": ["slot_1", "slot_2", ...]
    }
}

if intent = "correct_information":
{
    "intent": "correct_information",
    "slots": {
        "slot_name": value or empty string
        ...
    }
}

if intent = "end_conversation":
{
    "intent": "end_conversation",
    "slots": { }
}
"""

dm = """
You are the Dialogue Manager.
You are given the dialogue state, including turn_count, intent, filled slots, and recommended_cars.
 
ROLE:
- Decide the next best action by following the decision rules below in order — stop at the first rule that matches
 
AVAILABLE ACTIONS:
- slot_filling: select empty slots by confronting input slots with the full list below and put them into "value"
- correct_info: user is correcting a previously given value
- help_user: user is asking for your opinion or guidance (not a recommendation)
- recommend: recommend cars based on filled slots
- end_conversation: end the conversation
 
DECISION RULES:
- If intent == "provide_information" → slot_filling
- If intent == "end_conversation" → end_conversation
- If intent == "correct_information" → correct_info
- If intent == "require_information" → help_user
- If intent == "request_recommendation" → recommend

IMPORTANT:
The action field MUST be exactly one of:
slot_filling
correct_info
help_user
recommend
end_conversation

AVAILABLE SLOTS WITH DESCRIPTIONS:
- model_year: year of the car
- manufacturer: the company that produces the car
- model: the specific model of the car
- vehicle_class: the class/style of the car
- fuel_type: the type of fuel the car uses
- engine_displacement_l: engine displacement in liters
- cylinders: number of cylinders
- transmission: type of transmission
- drivetrain: drivetrain and traction system
- cargo_volume_cu_ft: cargo volume
- combined_mpg: combined fuel efficiency
- driving_range_miles: driving range
- electric_range_miles: electric range
- phev_blended_mpge: plug-in hybrid blended efficiency
- annual_fuel_cost_usd: annual fuel cost
- co2_emissions_g_per_mile: CO2 emissions
- fe_rating_1_10: fuel efficiency rating

OUTPUT FORMAT:
Respond with only a JSON object, no other text:
{
    "action": "slot_filling" | "correct_info" | "help_user" | "recommend" | "end_conversation",
    "value": <list of slot names to target for slot_filling, empty otherwise>
}
"""

nlg = """
ROLE:
- You are the Natural Language Generation (NLG) module.
- You are a friendly assistant having a genuine conversation with someone who is looking for a new car.
 
WHAT YOU HAVE:
- The next action to take
- The dialogue state, which includes turn_count, filled slots, and recommended_cars
- The conversation history contains the user's actual words. Use it as the primary context for your response.

CONVERSATION RULES:
- If action != recommend, ignore any previously recommended cars entirely.
- Do not maintain conversational continuity from prior recommendations.
- Treat new user input as a fresh direction unless slots explicitly overlap.
- Max 400 characters per response, be a little lengthy in your answers
- Never say that you have options available, or similar sentences that might point to a recommendation
- Never mention slots
- Do not use Markdown
- Do not restate extracted slots back to the user (only for the slot_filling action)
- Avoid unnecessary technical terminology unless the user introduced it.
- Do not offer to dive deeper into car specs unless discussing already-recommended cars
- Always react to what the user said before asking anything — acknowledge, empathize, or comment first
- Never suggest or describe a car type mid-conversation, even casually or as a starting point, wait for the recommend action

AVAILABLE ACTIONS WITH DESCRIPTIONS:
- slot_filling: make this action sound more like a follow-up sentence to the conversation, rather than a question to fill a slot
    Continue the conversation naturally while gathering information about the requested topic.
    The topic is only a conversational direction, never mention it explicitly.
    React to what the user already said before asking.
    Avoid sounding like a questionnaire.
- correct_info: acknowledge the change of the slot's value, without explicitly mentioning the slot itself
- help_user: user asks for your help, output a description of the options the user has available and your opinion on which one they should select
- recommend: present the recommended cars from the additional info provided
- end_conversation: wrap up the conversation
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
    model = InitModel(model_name, dtype="auto")
    model.eval()

    nlu_engine = NLU(model, tokenizer, prepare_text, nlu)
    dm_engine = DM(model, tokenizer, prepare_text, dm)
    nlg_engine = NLG(model, tokenizer, prepare_text, nlg)
    reco_engine = CarRecommender(top_k=3)

    messages = []
    dialogue_state = {
        "intent": None,
        "slots": {},
        "recommended_cars": [],
        "turn_count": 0
    }


    intro = (
        "Hi! I will be your assistant to find the best car that fits your needs."
        " We will discuss about your routine, passions and other aspects of your life,"
        " everything can be useful to give you the best results!"
    )
    print(f"Assistant: {intro}")
    messages.append({"role": "assistant", "content": intro})

    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        dialogue_state["turn_count"] += 1

        try:
            state = nlu_engine.parse(user_input, messages, args.n_exchanges)
        except Exception as e:
            print(f"NLU parsing failed: {e}")
            continue

        #print(f"[DEBUG] NLU: intent={state['intent']} | extracted={state.get('slots', {})}")

        dialogue_state["intent"] = state["intent"]
        intent = state["intent"]
        payload = state.get("slots", {})

        if intent in ("provide_information", "correct_information"):
            for slot, value in payload.items():
                if slot in SLOTS:
                    dialogue_state["slots"][slot] = value
        elif intent == "require_information":
            dialogue_state["required_slots"] = payload.get("required_slots", [])

        action, value = dm_engine.decide(dialogue_state)   # contract: returns (action_name, payload)

        #print(f"\n[DEBUG] Turn: {dialogue_state['turn_count']} | Action: {action} | Value: {value}")
        #print(f"[DEBUG] Slots: {json.dumps(dialogue_state['slots'], indent=2)}")
        #print()

        response = ""

        if action == "end_conversation":
            response = "Thank you for using our car recommendation service. If you have any more questions in the future, feel free to ask. Have a great day!"
            messages.append({"role": "assistant", "content": response})
            print(response)
            break

        elif action == "recommend":
            recs = reco_engine.recommend(dialogue_state["slots"])
            if not recs:
                response = "I'm sorry, I couldn't find any cars matching your preferences. Could you please provide more details or adjust your criteria?"
            else:
                def format_car(car):
                    # Only include fields that actually have a value in the CSV row
                    fields = {k: v for k, v in car.items() if v}
                    lines = [f"  {k}: {v}" for k, v in fields.items()]
                    return "\n".join(lines)

                extra = "\n\n".join(
                    f"Option {i+1}:\n{format_car(car)}"
                    for i, (car, score) in enumerate(recs)
                )
                
                # Track recommended cars for future reference
                dialogue_state["recommended_cars"] = [
                    {"index": i, "car": car, "score": score}
                    for i, (car, score) in enumerate(recs)
                ]
                
                response = nlg_engine.generate(action, dialogue_state, extra, messages, args.n_exchanges)
            messages.append({"role": "assistant", "content": response})

        elif action in {"slot_filling", "correct_info", "help_user", "follow_up"}:
            # Pass available values to prevent hallucinations
            response = nlg_engine.generate(action, dialogue_state, value, messages, args.n_exchanges)
            messages.append({"role": "assistant", "content": response})

        else:
            response = "I cannot assist with that. I'm here to help you find the perfect car."
            messages.append({"role": "assistant", "content": response})

        print(f"Assistant: {response}")

if __name__ == "__main__":
    args = parse_args()
    interact(args)