import os
import sys
from dotenv import load_dotenv

# Try to import the Google GenAI SDK
try:
    from google import genai
except ImportError:
    print("Error: The 'google-genai' library is not installed or not found.")
    print("Please run: pip install google-genai")
    sys.exit(1)

# Load environment variables
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

print(f"API Key found: {api_key[:5]}...{api_key[-5:]}")

# Test with the correct API structure
print("\n--- Testing Models ---")

# Pass the API key explicitly
client = genai.Client(api_key=api_key)
working_model = None

# Models to test (in order of preference)
candidates = [
    "models/gemini-2.5-flash",     # Latest and best
    "models/gemini-2.0-flash",     # Stable fallback
    "models/gemini-flash-latest"   # Generic latest
]

print("Testing models with correct API structure...")

for model in candidates:
    try:
        print(f"Trying {model}...", end=" ")

        response = client.models.generate_content(
            model=model,
            contents="Hello! Please respond with a simple greeting."
        )

        print("SUCCESS!")
        print(f"Response: {response.text}")
        working_model = model
        break  # Stop on first success

    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            print("Failed (Quota)")
        elif "404" in error_str:
            print("Failed (Not Found)")
        elif "403" in error_str:
            print("Failed (Forbidden)")
        else:
            print(f"Failed: {error_str}")

if working_model:
    print(f"\nSUCCESS: Functional model confirmed: {working_model}")

    # Test with a diabetes-related prompt similar to the app
    print(f"\n--- Testing Complex Diabetes Prompt ---")
    try:
        complex_prompt = """You are a medical AI assistant specializing in diabetes prevention and management.

Patient Risk Profile:
- Risk Category: Moderate Risk
- Risk Percentage: 65%
- Risk Factors: High BMI, Family history of diabetes

Current Recommendations:
- Regular glucose monitoring
- Dietary modifications

Based on this information, provide personalized, actionable advice for reducing diabetes risk.
Include specific lifestyle changes, dietary recommendations, exercise plans, and monitoring strategies.
Keep the response concise (under 500 words) and organized with clear sections."""

        complex_response = client.models.generate_content(
            model=working_model,
            contents=complex_prompt
        )

        print("Complex prompt SUCCESS!")
        print(f"Response: {complex_response.text}")

    except Exception as e:
        print(f"Complex prompt failed: {e}")
else:
    print("\nFAILED: All attempts failed.")
    print("Possible issues:")
    print("- API key quota exceeded")
    print("- API key invalid")
    print("- Network connectivity issues")
