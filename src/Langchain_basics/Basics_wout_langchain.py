import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


## STANDARD API CALL
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a limerick about the Python programming language.",
)

print(response.text)


## STRUCTURED OUTPUT (ex: calendar event)

from pydantic import BaseModel

# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# --------------------------------------------------------------
# Step 2: Call the model
# --------------------------------------------------------------

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Alice and Bob are going to a science fair on Friday.",
    config={
        "response_mime_type": "application/json",
        "response_schema": CalendarEvent,
    },
)

# --------------------------------------------------------------
# Step 3: Parse the response
# --------------------------------------------------------------

event = response.parsed

print(event.name)
print(event.date)
print(event.participants)


##Tools
import json
import requests
from google.genai import types
from pydantic import Field

# --------------------------------------------------------------
# 1. External tools (real Python functions)
# --------------------------------------------------------------


def geocode(city: str):
    """Simple geocoder using Open-Meteo (no API key needed)."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(url, params={"name": city, "count": 1})
    data = resp.json()

    if not data.get("results"):
        raise ValueError(f"Could not geocode city: {city}")

    result = data["results"][0]
    return {
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "name": result["name"],
        "country": result["country"],
    }


def get_weather(latitude: float, longitude: float):
    """Fetch current weather."""
    url = "https://api.open-meteo.com/v1/forecast"
    resp = requests.get(
        url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m",
        },
    )
    return resp.json()["current"]


# --------------------------------------------------------------
# 2. Output schema (structured response)
# --------------------------------------------------------------


class WeatherResponse(BaseModel):
    location: str = Field(description="City name")
    temperature: float = Field(description="Current temperature in Celsius")
    summary: str = Field(description="Natural language response")


# --------------------------------------------------------------
# 3. Define tools for Gemini
# --------------------------------------------------------------

geocode_tool = types.FunctionDeclaration(
    name="geocode",
    description="Get latitude and longitude for a city name.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "city": {"type": "STRING"},
        },
        "required": ["city"],
    },
)

weather_tool = types.FunctionDeclaration(
    name="get_weather",
    description="Get current weather using latitude and longitude.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "latitude": {"type": "NUMBER"},
            "longitude": {"type": "NUMBER"},
        },
        "required": ["latitude", "longitude"],
    },
)

# --------------------------------------------------------------
# 4. Step helper to run tool calls
# --------------------------------------------------------------


def run_tool(name, args):
    if name == "geocode":
        return geocode(**args)
    if name == "get_weather":
        return get_weather(**args)
    raise ValueError(f"Unknown tool: {name}")


# --------------------------------------------------------------
# 5. STEP 1 — Ask Gemini
# --------------------------------------------------------------

messages = [
    types.Content(
        role="user", parts=[types.Part(text="What's the weather in Brussels today?")]
    )
]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=messages,
    config=types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[geocode_tool, weather_tool])]
    ),
)

# --------------------------------------------------------------
# 6. STEP 2 — Handle tool calls (multi-step loop)
# --------------------------------------------------------------

tool_outputs = []

for fc in response.function_calls:
    result = run_tool(fc.name, fc.args)

    tool_outputs.append(
        types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=fc.name,
                    response=result,
                )
            ],
        )
    )

# --------------------------------------------------------------
# 7. STEP 3 — Send tool results back to Gemini
# --------------------------------------------------------------

final_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        messages[0],  # original user message
        response.candidates[0].content,  # model tool request
        *tool_outputs,  # tool results
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=WeatherResponse,
    ),
)

# --------------------------------------------------------------
# 8. Final parsed result
# --------------------------------------------------------------

result = final_response.parsed

print("Location:", result.location)
print("Temperature:", result.temperature)
print("Summary:", result.summary)


## Retrieval

# --------------------------------------------------------------
# Mock knowledge base tool (same as yours)
# --------------------------------------------------------------


def search_kb(question: str):
    with open("kb.json", "r") as f:
        return json.load(f)


# --------------------------------------------------------------
# Tool definition for Gemini
# --------------------------------------------------------------

kb_tool = types.FunctionDeclaration(
    name="search_kb",
    description="Get the answer to the user's question from the knowledge base.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "question": {"type": "STRING"},
        },
        "required": ["question"],
    },
)

# --------------------------------------------------------------
# Structured output schema (same as OpenAI Pydantic)
# --------------------------------------------------------------


class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")


system_prompt = (
    "You are a helpful assistant that answers questions from a "
    "knowledge base about our e-commerce store."
)

# ==============================================================
# STEP 1 — Ask Gemini (tool-enabled)
# ==============================================================

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            role="user", parts=[types.Part(text="What is the return policy?")]
        )
    ],
    config=types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[kb_tool])]
    ),
)

# --------------------------------------------------------------
# STEP 2 — Check if tool was called
# --------------------------------------------------------------

tool_outputs = []

if response.function_calls:
    for fc in response.function_calls:
        # run tool
        if fc.name == "search_kb":
            result = search_kb(**fc.args)

        tool_outputs.append(
            types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=fc.name,
                        response=result,
                    )
                ],
            )
        )
tool_outputs

# ==============================================================
# STEP 3 — Send tool result back (structured output)
# ==============================================================

final_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            role="user", parts=[types.Part(text="What is the return policy?")]
        ),
        response.candidates[0].content,
        *tool_outputs,
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=KBResponse,
    ),
)

parsed = final_response.parsed

print(parsed.answer)
print(parsed.source)
