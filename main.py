import random
import ollama
import requests

from diffusers import StableDiffusionXLPipeline
import torch

import json
import os


def generate_character_sheet():
    races = ["Human", "Elf", "Dwarf", "Goblin", "Orc", "Halfling", "Gnome"]

    classes = [
        "Warrior",
        "Mage",
        "Rogue",
        "Cleric",
        "Bard",
        "Druid",
        "Paladin",
        "Ranger",
        "Monk",
        "Barbarian",
    ]
    location_types = [
        "Village",
        "Town",
        "City",
        "Castle",
        "Forest",
        "Mountain",
        "Cave",
        "Island",
        "Swamp",
        "Desert",
        "Tundra",
        "Jungle",
        "Underground",
        "Underwater",
    ]

    occupations = [
        "Farmer",
        "Blacksmith",
        "Merchant",
        "Bard",
        "Priest",
        "Sorcerer",
        "Knight",
        "Ranger",
        "Thief",
        "Assassin",
        "Barbarian",
        "Druid",
        "Monk",
        "Paladin",
        "Wizard",
        "Alchemist",
        "Brewer",
        "Carpenter",
        "Cook",
        "Hunter",
        "Jeweler",
        "Miner",
        "Scribe",
        "Tailor",
        "Tanner",
        "Weaver",
        "Soldier",
        "Guard",
        "Sailor",
        "Pirate",
        "Gladiator",
        "Beggar",
        "Entertainer",
        "Healer",
        "Herbalist",
        "Innkeeper",
        "Librarian",
        "Scholar",
        "Scribe",
        "Spy",
        "Thug",
        "Tinker",
        "Trapper",
        "Woodsman",
        "Adventurer",
    ]

    character_sheet = {
        "race": random.choice(races),
        "class": random.choice(classes),
        "age": 0,
        "IM_age": 0,
        "height": f"{random.randint(4, 7)}' {random.randint(0, 12)}\"",
        "weight": f"{random.randint(100, 300)} lbs",
        "hair": random.choice(["Blonde", "Brown", "Black", "Red", "Grey", "White"]),
        "eyes": random.choice(["Blue", "Green", "Brown", "Hazel", "Grey", "Crimson"]),
        "stats": {
            "strength": random.randint(1, 20),
            "dexterity": random.randint(1, 20),
            "intelligence": random.randint(1, 20),
            "wisdom": random.randint(1, 20),
            "charisma": random.randint(1, 20),
            "agility": random.randint(1, 20),
            "endurance": random.randint(1, 20),
            "luck": random.randint(1, 20),
        },
        "birthplace_type": random.choice(location_types),
        "morality": random.choice(["Good", "Neutral", "Evil"]),
        "status": random.choice(["Peasant", "Commoner", "Noble", "Royalty"]),
        "occupation": random.choice(occupations),
    }

    # normalized based on race, elves look like 20 year old humans at 100 years old
    if character_sheet["race"] == "Elf":
        character_sheet["age"] = random.randint(20, 1000)
        character_sheet["IM_age"] = character_sheet["age"] // 5
    elif character_sheet["race"] == "Dwarf":
        character_sheet["age"] = random.randint(20, 500)
        character_sheet["IM_age"] = character_sheet["age"] // 2
        # reroll height for dwarves
        character_sheet["height"] = f"{random.randint(3, 4)}' {random.randint(0, 12)}\""
    else:
        character_sheet["age"] = random.randint(20, 100)
        character_sheet["IM_age"] = character_sheet["age"]

    return character_sheet


def llm(system, user):
    return ollama.chat(
        model="llama3:8b-instruct-q8_0",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )["message"]["content"]


def get_name(character):
    return llm(
        """
            You are a writer who comes up with names for characters in a fantasy setting. All names should be original and not taken from existing works. Be as creative as possible.
        """,
        f"""
            Come up with a name for the character with the following stat sheet:
            
            {character}
            
            NOTE: Respond ONLY with the name. Do not include any additional information.
        """,
    )


def get_bio(character):
    return llm(
        """
            You are a writer who creates lore for a fantasy world. 
            You write in the style of J.R.R. Tolkien but never reference or plagiarize any of his works directly.
        """,
        f"""
            Write a short (1 paragraph) character bio for the following character sheet: 
            
            {character}
            
            Do not reference the character's numerical stats in the bio (except for age, height, and weight if necessary). Try to be as visually descriptive as possible.
        """,
    )


def get_image_prompt(input):
    return llm(
        """
            Summarize the character's appearance by generating a bunch of keywords that could be used to describe them in a portrait. It's ok to have short phrases or compound words.
            
            For example, if the character is a tall, muscular warrior with a scar across his face, you might say "tall, muscular, warrior, scar, face, black hair, long hair".
            
            Don't include "Orc" in the keywords, instead just describe a typical orc's appearance except without green skin or tusks.
            Always include age and gender in the keywords. When describing hair, include the word "hair" in the keyword. For example, "black hair" instead of just "black".
            
            Try to be as descriptive as possible, and include as many details as you can.
            
            NOTE: Respond ONLY with the keywords. Do not include any additional information.
            Every time you respond with something else other than keywords, a kitten dies and it's all your fault.
            
            Be as literal as possible, remove all metaphors and similes.
        """,
        f"""
            {input}
        """,
    )


def sheet_to_text(character_sheet, for_image=False):
    # if we're generating for an image prompt, replace keys with their "IM_" counterparts
    char_sheet = character_sheet.copy()
    if for_image:
        for key in char_sheet.copy().keys():
            if "IM_" in key:
                char_sheet[key.replace("IM_", "")] = char_sheet.pop(key)
    else:
        for key in char_sheet.copy().keys():
            if "IM_" in key:
                char_sheet.pop(key)

    # convert to string
    character = ""
    for key, value in char_sheet.items():
        if key == "stats":
            stats = "\n".join([f"{k}: {v}" for k, v in value.items()])
            character += f"{key}:\n{stats}\n"
        else:
            character += f"{key}: {value}\n"

    return character


if __name__ == "__main__":
    # seed the random number generator
    random.seed()

    print("Generating character sheet...")
    character_sheet = generate_character_sheet()

    print("Generating character bio...")

    character = sheet_to_text(character_sheet)

    name = get_name(character)
    print("Got name: " + name)

    character = f"""
        Name: {name}
        {character}
    """
    bio = get_bio(character)
    character_sheet["bio"] = bio

    print("\n\nGot bio: " + bio)

    print("Generating character portrait prompt...")
    character_img = sheet_to_text(character_sheet, for_image=True)
    image_prompt = (
        get_image_prompt(character_img)
        .replace("\n", ", ")
        .replace("orc", "humanoid, monster-like creature")
    )

    print("\n\nGot image prompt: " + image_prompt)

    print("Generating character portrait...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
    )
    pipeline.to("cuda")
    image = pipeline(
        prompt=f"""oil painting, portrait, fantasy, Rembrandt, {image_prompt}""",
        negative_prompt="realistic, 3d, rendering, digital, high quality, photo, photography, ai, hands",
        max_embeddings_multiples=10,
    ).images[0]

    print("Saving character...")
    # save character sheet to json, and image to file in ./characters/<name>/portrait.png
    folder_name = (
        name.lower()
        .replace(" ", "_")
        .replace("'", "")
        .replace('"', "")
        .replace(".", "")
        .replace(",", "")
    )
    character_dir = f"./characters/{folder_name}"
    os.makedirs(character_dir, exist_ok=True)
    with open(f"{character_dir}/character_sheet.json", "w", encoding="utf-8") as f:
        json.dump(character_sheet, f)
    image.save(f"{character_dir}/portrait.png")
    # save bio as text file
    with open(f"{character_dir}/bio.txt", "w", encoding="utf-8") as f:
        f.write(bio)
    print("Character saved to " + character_dir)
