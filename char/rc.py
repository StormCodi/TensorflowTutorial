from PIL import Image
import os

input_dir = r'C:/Users/Minec/OneDrive/Documents/Coding Projects/Tensoflow course/char/Bmp'
output_dir = r'C:/Users/Minec/OneDrive/Documents/Coding Projects/Tensoflow course/char/Resized'

# set target image size
size = (28, 28)

# loop over all subdirectories in the input directory
for sub_dir_name in os.listdir(input_dir):
    sub_dir_path = os.path.join(input_dir, sub_dir_name)
    # check if the subdirectory is a directory and not a file
    if os.path.isdir(sub_dir_path):
        # loop over all BMP files in the subdirectory
        for file_name in os.listdir(sub_dir_path):
            if file_name.endswith('.png'):
                # open the image
                img_path = os.path.join(sub_dir_path, file_name)
                with Image.open(img_path) as img:
                    # resize the image
                    img = img.resize(size)
                    # convert to grayscale
                    img = img.convert('L')
                    # save the resized and converted image to the output directory
                    output_path = os.path.join(output_dir, sub_dir_name, file_name)
                    try:
                        img.save(output_path)
                    except Exception as e:
                        print(f"Error saving {file_name}: {e}")
