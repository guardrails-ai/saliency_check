import pytest
from guardrails import Guard
from pydantic import BaseModel, Field
from validator import SaliencyCheck


# Create a pydantic model with a field that uses the custom validator
class ValidatorTestObject(BaseModel):
    text: str = Field(
        validators=[
            SaliencyCheck(
                "tests/assets/",
                llm_callable="gpt-3.5-turbo",
                threshold=0.25,
                on_fail="exception",
            )
        ]
    )


# Create the guard object
guard = Guard.from_pydantic(output_class=ValidatorTestObject)


# Test happy path
@pytest.mark.parametrize(
    "value",
    [
        """
        {
            "text": "San Francisco is a major Californian city, known for its finance, culture, and density. Originally inhabited by the Yelamu tribe, the city grew rapidly during the Gold Rush and became a major West Coast port. Despite a devastating earthquake and fire in 1906, San Francisco rebuilt and played significant roles in World War II and international relations. The city is also known for its liberal activism and social movements."
        }
        """,
    ],
)
def test_happy_path(value):
    """Test happy path."""
    response = guard.parse(value)
    print("Happy path response", response)
    assert response.validation_passed is True


# Test fail path
@pytest.mark.parametrize(
    "value",
    [
        """
        {
            "text": "Boston in Massachusetts is a city in the United States. It is the 21st most populous city in the United States, and the third most populous in Massachusetts."
        }
        """,
    ],
)
def test_fail_path(value):
    """Test fail path."""
    with pytest.raises(Exception):
        response = guard.parse(
            value,
        )
        print("Fail path response", response)
