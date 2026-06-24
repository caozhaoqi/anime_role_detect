#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import os
import requests
import argparse
import logging
from PIL import Image
from io import BytesIO

# 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_characters")


def generate_character_image(character_name, output_dir, num_images=5):
    """

    Args:
        character_name: 
        output_dir: 
        num_images: 
    """
    os.makedirs(output_dir, exist_ok=True)

    # 
    character_descriptions = {
        "arona": "Arona from Blue Archive, blue short hair, halo, school uniform, cute, anime style",
        "plana": "Plana from Blue Archive, black long hair, halo, school uniform, elegant, anime style",
    }

    if character_name not in character_descriptions:
        logger.error(f": {character_name}")
        return

    description = character_descriptions[character_name]
    logger.info(f" {character_name} ...")

    for i in range(num_images):
        try:
            # Trae API
            prompt = f"{description}, high quality, detailed, anime, 4k"
            url = f"https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt={prompt}&image_size=portrait_4_3"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 
            image = Image.open(BytesIO(response.content))
            output_path = os.path.join(output_dir, f"{character_name}_{i+1}.png")
            image.save(output_path)
            logger.info(f": {output_path}")

        except Exception as e:
            logger.error(f": {e}")
            continue

    logger.info(f"{character_name} ")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--character",
        type=str,
        choices=["arona", "plana", "both"],
        default="both",
        help="",
    )
    parser.add_argument("--num-images", type=int, default=5, help="")
    parser.add_argument("--output-dir", type=str, default="data/generated", help="")

    args = parser.parse_args()

    if args.character == "both" or args.character == "arona":
        generate_character_image(
            "arona", os.path.join(args.output_dir, "_"), args.num_images
        )

    if args.character == "both" or args.character == "plana":
        generate_character_image(
            "plana", os.path.join(args.output_dir, "_"), args.num_images
        )


if __name__ == "__main__":
    main()
