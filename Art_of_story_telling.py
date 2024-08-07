import transformers
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import openai
import os
from dotenv import load_dotenv  # Import for dotenv

# Load environment variables from .env file (assuming it's in the same directory)
load_dotenv()

# Access OpenAI key securely from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def get_and_preprocess_image(path_or_url):
    image = Image.open(path_or_url)
    inputs = processor(images=image, return_tensors="pt")
    return inputs, image


def detect_objects(inputs, model, image, processor):
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    detected_objects = [model.config.id2label[label.item()] for label in results["labels"]]
    return detected_objects


def generate_story(objects):
    prompt = f"The image contains {', '.join(objects)}. Please write a story."

    # Generate story using OpenAI API
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,  # Adjust max_tokens for story length per image
        temperature=0.3
    )
    return response.choices[0].text.strip()


def main():
    # Get number of images
    num_images = int(input("Enter the number of images you want to upload: "))

    all_detected_objects = []  # List to store all detected objects
    all_stories = []  # List to store all stories

    # Use a single model instance (memory efficient)
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    for image_index in range(num_images):
        print(f"\n**Image {image_index + 1}**")

        # Get image path/URL
        image_path_or_url = input(f"Enter path or URL of image {image_index + 1}: ")

        # Process image
        image_inputs, image_processed_image = get_and_preprocess_image(image_path_or_url)

        print("\n**Detected Objects:**")
        detected_objects = detect_objects(image_inputs, model, image_processed_image, processor)
        print(", ".join(detected_objects))

        all_detected_objects.extend(detected_objects)  # Add to all objects list

        print("\n**Story for Image:**")
        story = generate_story(detected_objects)
        print(story)
        all_stories.append(story)  # Add to all stories list

    # Combine detected objects (if desired)
    combined_objects = list(set(all_detected_objects))  # Remove duplicates

    # Generate story based on combined objects (optional)
    if combined_objects:  # Check if any objects were detected
        print("\n**Combined Objects:**")
        print(", ".join(combined_objects))

        combined_story_prompt = f"The images contain a combination of {', '.join(combined_objects)}. Please write a story that incorporates these elements."
        combined_story = generate_story(combined_story_prompt)
        print("\n**Combined Story:**")
        print(combined_story)
    else:
        print("\n**No objects detected in any images.")


if __name__ == "__main__":
    main()
