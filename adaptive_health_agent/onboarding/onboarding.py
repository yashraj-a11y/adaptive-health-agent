"""
Onboarding Module

Guides new users through an interactive onboarding flow to collect
identity, health context, goals, communication preferences, and
emergency contact information. Produces an initial Living Profile.
"""

import sys
from memory.living_profile import create_profile


def run_onboarding() -> dict:
    """Run the interactive onboarding flow for a new user.

    Asks questions in a fixed order, one at a time, and builds
    a user_data dict that is passed to create_profile().

    Returns:
        dict: The initial Living Profile created from onboarding data.
    """
    print("\n" + "=" * 50)
    print("  Welcome to the Adaptive Health Agent")
    print("  Let's get to know you so I can personalize")
    print("  your health monitoring experience.")
    print("=" * 50 + "\n")

    user_data = {}

    # 1. Name
    name = input("1. What's your name? ").strip()
    user_data["name"] = name
    user_data["user_id"] = name.lower().replace(" ", "_")

    # 2. Age
    while True:
        age_str = input("\n2. How old are you? ").strip()
        try:
            age = int(age_str)
            if 1 <= age <= 120:
                user_data["age"] = age
                break
            else:
                print("   Please enter a valid age (1-120).")
        except ValueError:
            print("   Please enter a number.")

    # 3. Activity level
    print("\n3. What's your typical activity level?")
    print("   1. Sedentary (mostly sitting, little exercise)")
    print("   2. Lightly active (light exercise 1-3 days/week)")
    print("   3. Moderately active (moderate exercise 3-5 days/week)")
    print("   4. Very active (hard exercise 6-7 days/week)")
    while True:
        activity_choice = input("   Choose (1-4): ").strip()
        if activity_choice in ("1", "2", "3", "4"):
            activity_levels = {
                "1": "sedentary",
                "2": "lightly_active",
                "3": "moderately_active",
                "4": "very_active"
            }
            user_data["activity_level"] = activity_levels[activity_choice]
            break
        else:
            print("   Please choose 1, 2, 3, or 4.")

    # 4. Known health conditions
    print("\n4. Do you have any known health conditions?")
    conditions_str = input("   (type 'none' if none, or list separated by commas): ").strip()
    if conditions_str.lower() == "none":
        user_data["known_conditions"] = []
    else:
        user_data["known_conditions"] = [c.strip() for c in conditions_str.split(",") if c.strip()]

    # 5. Current medications
    print("\n5. Are you taking any medications that affect heart rate, sleep, or stress?")
    meds_str = input("   (type 'none' if none, or list separated by commas): ").strip()
    if meds_str.lower() == "none":
        user_data["medications"] = []
    else:
        user_data["medications"] = [m.strip() for m in meds_str.split(",") if m.strip()]

    # 6. Main goal
    print("\n6. What's your main goal with the health agent?")
    print("   1. Improve sleep quality")
    print("   2. Manage stress better")
    print("   3. Optimize fitness and recovery")
    print("   4. Monitor a health condition")
    print("   5. General wellness improvement")
    print("   6. Just monitoring — notify me only if something is off")
    while True:
        goal_choice = input("   Choose (1-6): ").strip()
        if goal_choice in ("1", "2", "3", "4", "5", "6"):
            goals = {
                "1": "improve_sleep",
                "2": "manage_stress",
                "3": "optimize_fitness",
                "4": "monitor_condition",
                "5": "general_wellness",
                "6": "monitoring_only"
            }
            user_data["goals"] = [goals[goal_choice]]
            break
        else:
            print("   Please choose 1-6.")

    # 7. Communication style
    print("\n7. How would you like me to communicate with you?")
    print("   1. Clinical and data-driven — give me the numbers")
    print("   2. Balanced — mix of data and plain language")
    print("   3. Casual and friendly — keep it conversational")
    print("   4. Minimal — only alert me when something is important")
    while True:
        style_choice = input("   Choose (1-4): ").strip()
        if style_choice in ("1", "2", "3", "4"):
            style_map = {
                "1": {"style": "clinical", "directness": 1, "depth": 1, "tone": 1, "length": 1, "framing": 1},
                "2": {"style": "balanced", "directness": 3, "depth": 3, "tone": 3, "length": 3, "framing": 3},
                "3": {"style": "casual", "directness": 5, "depth": 4, "tone": 5, "length": 4, "framing": 5},
                "4": {"style": "minimal", "directness": 3, "depth": 4, "tone": 3, "length": 5, "framing": 3},
            }
            style = style_map[style_choice]
            user_data["communication_style"] = style["style"]
            user_data["directness"] = style["directness"]
            user_data["depth"] = style["depth"]
            user_data["tone"] = style["tone"]
            user_data["length"] = style["length"]
            user_data["framing"] = style["framing"]

            # Set alert sensitivity based on goal
            if user_data["goals"][0] == "monitoring_only":
                user_data["alert_sensitivity"] = "low"
            elif user_data["goals"][0] == "monitor_condition":
                user_data["alert_sensitivity"] = "high"
            else:
                user_data["alert_sensitivity"] = "normal"

            break
        else:
            print("   Please choose 1-4.")

    # 8. Emergency contact
    print("\n8. Would you like to set up an emergency contact?")
    print("   This person would be notified only in Level 5 emergencies.")
    ec_choice = input("   (yes/no): ").strip().lower()
    if ec_choice in ("yes", "y"):
        ec_name = input("   Emergency contact name: ").strip()
        ec_contact = input("   Emergency contact method (phone/email): ").strip()
        user_data["emergency_contact"] = {
            "name": ec_name,
            "contact": ec_contact
        }
    else:
        user_data["emergency_contact"] = None

    # Set default engagement times and patterns
    user_data["best_engagement_times"] = []
    user_data["engagement_patterns"] = "unknown"

    # Create the Living Profile
    profile = create_profile(user_data)

    # Print closing message
    print("\n" + "=" * 50)
    print(f"  Welcome, {name}! Your health profile is set up.")
    print(f"  Baseline status: LEARNING")
    print(f"  I'll spend the first {14} days learning your")
    print(f"  personal baselines before making assessments.")
    print(f"  You can talk to me anytime — just type a message.")
    print("=" * 50 + "\n")

    return profile
