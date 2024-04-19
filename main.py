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
        "IM_visible_age": 0,
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
        character_sheet["visible_age"] = character_sheet["age"] // 5
    elif character_sheet["race"] == "Dwarf":
        character_sheet["age"] = random.randint(20, 500)
        character_sheet["visible_age"] = character_sheet["age"] // 2
    else:
        character_sheet["age"] = random.randint(20, 100)
        character_sheet["visible_age"] = character_sheet["age"]

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
            You are a writer who comes up with names for characters in an RPG game. All names should be original and not taken from existing works.
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
            You are a writer who creates lore for an RPG game. 
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
            Summarize the character's appearance in a way that would be suitable for an artist to create a portrait.
            Always include age, and other physical attributes in the description.
            
            If the input contains any race information, such as "Elf", "Dwarf", etc., replace it with a generic description of the character's appearance.
            Generally, elves have pointed ears, dwarves are short and stocky, goblins are small and have wrinkled skin, orcs are large and muscular, halflings are short and chubby, and gnomes are small and have pointy hats.
            
            NOTE: Respond ONLY with the description. Do not include any additional information.
            Do NOT write "Here's a prompt for an artist to create a portrait of the character" or anything similar.
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
    image_prompt = get_image_prompt(character_img)

    print("\n\nGot image prompt: " + image_prompt)

    print("Generating character portrait...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
    )
    pipeline.to("cuda")
    image = pipeline(
        prompt=f"""
        Style: oil painting, portrait, fantasy, renaissance, impressionist
        
        Visual description:
        {image_prompt}
        
        Bio:
        {bio}
        """,
        max_embeddings_multiples=3,
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
