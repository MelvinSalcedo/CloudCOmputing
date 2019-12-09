import glob
import os
from PIL import Image


def make_image_thumbnail(filename):
    # The thumbnail will be named "<original_filename>_thumbnail.jpg"
    base_filename, file_extension = os.path.splitext(filename)
    thumbnail_filename = f"{base_filename}_thumbnail{file_extension}"

    # Create and save thumbnail image
    image = Image.open(filename)
    image.thumbnail(size=(128, 128))
    image.save(thumbnail_filename, "JPEG")

    return thumbnail_filename


# Loop through all jpeg files in the folder and make a thumbnail for each
for image_file in glob.glob("*.jpg"):
    thumbnail_file = make_image_thumbnail(image_file)

    print(f"A thumbnail for {image_file} was saved as {thumbnail_file}")