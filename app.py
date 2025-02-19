from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Set up logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"

# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""

    description: str = Field(description="Raw description of the desired outcome")
    is_physical_outcome: bool = Field(
        description="Whether this text describes an outcome related to physical fitness"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class ExerciseDetails(BaseModel):
    """Second LLM call: Parse specific exercise details"""

    name: str = Field(description="Name of the exercise")
    description: str = Field(
        description="Suggested weight and reps or duration for the exercise"
    )

class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )


# --------------------------------------------------------------
# Step 2: Define the functions
# --------------------------------------------------------------


def extract_outcome_info(user_input: str) -> EventExtraction:
    """First LLM call to determine if input is a desired physical outcome of an exercise"""
    logger.info("Starting outcome extraction analysis")
    logger.debug(f"Input text: {user_input}")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Analyze if the text describes the intended outcome of a physical exercise.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=EventExtraction,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Extraction complete - Is outcome related to physical fitness: {result.is_physical_outcome}, Confidence: {result.confidence_score:.2f}"
    )
    return result


def get_exercise_rec(description: str) -> ExerciseDetails:
    """Second LLM call to determine the exercise"""
    logger.info("Starting to find an exercise")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Find one type of physical exercise to achieve the intended outcome.",
            },
            {"role": "user", "content": description},
        ],
        response_format=ExerciseDetails,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Parsed desired outcome - Exercise name: {result.name}"
    )
    return result


def generate_confirmation(exercise_details: ExerciseDetails) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Generate a natural response to achieve the intended outcome via a specific physical exercise and a suggestion for the weight and reps or length of time.",
            },
            {"role": "user", "content": str(exercise_details.model_dump())},
        ],
        response_format=EventConfirmation,
    )
    result = completion.choices[0].message.parsed
    logger.info("Confirmation message generated successfully")
    return result


# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------


def process_desired_outcome(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain with gate check"""
    logger.info("Processing desired outcome")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_outcome_info(user_input)

    # Gate check: Verify if it's an outcome related to physical fitness with sufficient confidence
    if (
        not initial_extraction.is_physical_outcome
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_physical_outcome: {initial_extraction.is_physical_outcome}, confidence: {initial_extraction.confidence_score:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with exercise processing")

    # Second LLM call: Get detailed exercise information
    exercise_details = get_exercise_rec(initial_extraction.description)

    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(exercise_details)

    logger.info("Physical exercise confirmation generated successfully")
    return confirmation


# --------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------

user_input = input("What is your desired physical outcome? ")

result = process_desired_outcome(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
else:
    print("This doesn't appear to be a desired physical outcome.")


# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

user_input = input("What is your desired physical outcome? ")

result = process_desired_outcome(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
else:
    print("This doesn't appear to be related to improving physical fitness.")
