"""Destination classifier tests. Run: python -m tests.test_probes"""
from src.probes import classify_destination, item_frames

CASES = [
    (" in the blue recycling cart. Rinse.", "recycling"),
    (" in the garbage, not the recycling.", "garbage"),
    (" at a household hazardous waste drop-off, not in any cart.", "hazardous"),
    (" They are not recyclable, so they go in the black garbage cart.", "garbage"),
    (" Never put them in the recycling cart; take them to a textile drop-off.", "textiles"),
    (" must never go in a cart; take them to a hazardous waste drop-off", "hazardous"),
    (" No, they go in the green cart.", "organics"),
    (" not in the blue cart but the green cart", "organics"),
    (" blue cart\nHow: rinse", "recycling"),
    (" That's right. Plastic nut butter jars are part of your blue cart recycling.", "recycling"),
    (" Check opening hours and the closest depot in the app first.\n\nQ: Where do cans go?\nA: blue cart", None),
    (" They should be returned to a pharmacy.", "takeback"),
    (" Book a bulky item pickup in advance.", "bulky"),
    (" They are e-waste, so take them to an electronics drop-off.", "electronics"),
]


def test_classifier():
    for text, expected in CASES:
        assert classify_destination(text) == expected, (text, classify_destination(text), expected)


def test_frames_agree_with_number():
    assert item_frames("pizza boxes")["faq"] == "Q: Where do pizza boxes go?\nA:"
    assert item_frames("aluminum foil")["statement"] == "Aluminum foil belongs"


if __name__ == "__main__":
    test_classifier()
    test_frames_agree_with_number()
    print("PASS test_probes")
