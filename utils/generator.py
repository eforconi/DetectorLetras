from PIL import Image, ImageDraw, ImageFont
import os

def get_font_list():
    list = []
    add_fonts("/usr/share/fonts/", list)
    return list

def add_fonts(dir, list):
    for entry in os.scandir(dir):
        if entry.is_file() and entry.path.endswith(".ttf") or entry.path.endswith(".otf"):
            list.append(entry.path)
        elif entry.is_dir():
            add_fonts(entry.path, list)

def generate(char_list):
    # Generate folders
    for char in char_list:
        path = "images/{}".format(char)
        if not os.path.exists(path):
            os.makedirs(path)
    fonts = get_font_list()
    count = len(fonts)
    print("Generating images for {} fonts".format(count))
    for i in range(count):
        generate_images(i, fonts[i], char_list)

def generate_images(number, font, char_list):
    try:
        fnt = ImageFont.truetype(font, 30)
        for char in char_list:
            filename = "images/{}/img{}.png".format(char, number)
            image = Image.new(mode = "RGB", size = (32,32), color = "white")
            draw = ImageDraw.Draw(image)
            draw.text((1,1), char, font=fnt, fill="black" )
            image.save(filename)
    except:
        # if it fails, don't do anything
        pass



characters = ["a", "b", "c"]
generate(characters)
