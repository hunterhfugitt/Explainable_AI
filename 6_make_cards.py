import json
from PIL import Image, ImageDraw, ImageFont
import requests
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

import PIL
import io
import urllib
import json
import pickle

def getImage(path):
   return OffsetImage(plt.imread(path, format="jpg"), zoom=.1)

def add_logo(background,foreground):
    bg_w, bg_h = background.size
    img_w, img_h = foreground.size
    img_offset = (50, (bg_h - img_h) // 4)
    background.paste(foreground, img_offset, foreground)
    return background

def add_color(image,c,transparency):
    color = Image.new('RGB',image.size,c)
    mask = Image.new('RGBA',image.size,(0,0,0,transparency))
    return Image.composite(image,color,mask).convert('RGB')

def write_image(background,color,text1,text2,foreground=''):
    background = add_color(background,color['c'],25)
    if not foreground:
        add_text(background,color,text1,text2)
    else:
        add_text(background,color,text1,text2,logo=True)
        add_logo(background,foreground)
    return background

def center_text(img,font,text1,text2,fill1,fill2):
    draw = ImageDraw.Draw(img) # Initialize drawing on the image
    w,h = img.size # get width and height of image
    t1_width, t1_height = draw.textsize(text1, font) # Get text1 size
    t2_width, t2_height = draw.textsize(text2, font) # Get text2 size
    p1 = ((w-t1_width)//7,h // 16) # H-center align text1
    p2 = ((w-t2_width)//7,h // 3 + h // 5) # H-center align text2
    draw.text(p1, text1, fill=fill1, font=font) # draw text on top of image
    draw.text(p2, text2, fill=fill2, font=font) # draw text on top of image
    return img

def add_text(img,color,text1,text2,logo=False,font='arial.ttf',font_size_to_use=500):
    draw = ImageDraw.Draw(img)
    p_font = color['p_font']
    s_font = color['s_font']
     
    # starting position of the message
    img_w, img_h = img.size
    height = img_h // 15.5
    font = ImageFont.truetype("arial.ttf", 20)
    #font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")
 
    if logo == False:
        center_text(img,font,text1,text2,p_font,s_font,font_size_to_use = 1000)
    else:
        text1_offset = (img_w // 10, height)
        text2_offset = (img_w // 10, 10*height)
        draw.text(text1_offset, text1, fill=p_font, font=font, font_whsize = 4* font_size_to_use)
        font = ImageFont.truetype("arial.ttf", 12)
        draw.text(text2_offset, text2, fill=s_font, font=font, font_size = 2*font_size_to_use)
    return img

colors = {
        'dark_blue':{'c':(27,53,81),'p_font':'rgb(255,255,255)','s_font':'rgb(255, 212, 55)'},
        'grey':{'c':(70,86,95),'p_font':'rgb(255,255,255)','s_font':'rgb(93,188,210)'},
        'light_blue':{'c':(93,188,210),'p_font':'rgb(27,53,81)','s_font':'rgb(255,255,255)'},
        'blue':{'c':(23,114,237),'p_font':'rgb(255,255,255)','s_font':'rgb(255, 255, 255)'},
        'orange':{'c':(242,174,100),'p_font':'rgb(0,0,0)','s_font':'rgb(0,0,0)'},
        'purple':{'c':(114,88,136),'p_font':'rgb(255,255,255)','s_font':'rgb(255, 212, 55)'},
        'red':{'c':(255,0,0),'p_font':'rgb(0,0,0)','s_font':'rgb(0,0,0)'},
        'yellow':{'c':(255,255,0),'p_font':'rgb(0,0,0)','s_font':'rgb(27,53,81)'},
        'yellow_green':{'c':(232,240,165),'p_font':'rgb(0,0,0)','s_font':'rgb(0,0,0)'},
        'green':{'c':(65, 162, 77),'p_font':'rgb(217, 210, 192)','s_font':'rgb(0, 0, 0)'},
        'black':{'c':(15, 15, 15),'p_font':'rgb(255, 255, 255)','s_font':'rgb(255, 255, 255)'},
        'white':{'c':(217, 210, 192),'p_font':'rgb(0, 0, 0)','s_font':'rgb(0, 0, 0)'}
        }


def _return_image(color,name,text,image_data,check2):
    dir = os.getcwd()+'\\static\\images\\card_images'
    # if(os.path.exists(fr'{dir}\\{name}.jpeg' and check2)):
    #     pass
    if(check2):
        pass
    else:
        colors = {
            'dark_blue':{'c':(27,53,81),'p_font':'rgb(255,255,255)','s_font':'rgb(255, 212, 55)'},
            'grey':{'c':(70,86,95),'p_font':'rgb(255,255,255)','s_font':'rgb(93,188,210)'},
            'light_blue':{'c':(93,188,210),'p_font':'rgb(27,53,81)','s_font':'rgb(255,255,255)'},
            'blue':{'c':(23,114,237),'p_font':'rgb(255,255,255)','s_font':'rgb(255, 255, 255)'},
            'orange':{'c':(242,174,100),'p_font':'rgb(0,0,0)','s_font':'rgb(0,0,0)'},
            'purple':{'c':(114,88,136),'p_font':'rgb(255,255,255)','s_font':'rgb(255, 212, 55)'},
            'red':{'c':(255,0,0),'p_font':'rgb(0,0,0)','s_font':'rgb(0,0,0)'},
            'yellow':{'c':(255,255,0),'p_font':'rgb(0,0,0)','s_font':'rgb(27,53,81)'},
            'yellow_green':{'c':(232,240,165),'p_font':'rgb(0,0,0)','s_font':'rgb(0,0,0)'},
            'green':{'c':(65, 162, 77),'p_font':'rgb(217, 210, 192)','s_font':'rgb(0, 0, 0)'},
            'black':{'c':(15, 15, 15),'p_font':'rgb(255, 255, 255)','s_font':'rgb(255, 255, 255)'},
            'white':{'c':(217, 210, 192),'p_font':'rgb(0, 0, 0)','s_font':'rgb(0, 0, 0)'}
            }  
        color = colors[color]
        check = True
        print('made_it_here')
        try:
            response = requests.get(image_data, verify=False, stream = True)    
        except(OSError, KeyError) as Error:
            check = False
        if(check is True):
            image = PIL.Image.open(io.BytesIO(response.content))
            img = image.convert("RGB")
            img.save(fr'{dir}\\{name}.jpeg')
        else:
            try:
                text_parsed = name.replace(' ', '+')
                page = requests.get('https://www.tcgplayer.com/search/magic/product?Language=English&productLineName=magic&q=' + text_parsed)                
                soup = BeautifulSoup(page.content, "html.parser")
                results = soup.find_all("div", {"class": 'lazy-image__wrapper'})[0]
                relevant = results['src']
                response = requests.get(relevant, verify=False)
                image_bytes = io.BytesIO(response.content)
                img = PIL.Image.open(image_bytes).convert("RGB")
                img = img.save(fr'{dir}\\{name}.jpeg')
            except:
                print("didn't work")
                path= os.getcwd()+'\\static\\images\\logo.jpg'
                foreground = Image.open(path).convert("RGBA")
                path= os.getcwd()+'\\static\\images\\blank card.jpg'
                background = Image.open(path)
                try:
                    background = write_image(background,color,name,text,foreground=foreground)
                    background.save(fr'{dir}\\{name}.jpeg')
                except:
                    replaced_name = ''.join(get_replacement_char(char) or char for char in name)
                    replaced_text = ''.join(get_replacement_char(char) or char for char in text)
                    background = Image.open(path)
                    background = write_image(background,color,replaced_name,replaced_text,foreground=foreground)
                    background.save(fr'{dir}\\{name}.jpeg')


def get_replacement_char(char):
    try:
        encoded_char = char.encode('latin-1')
        return encoded_char.decode('latin-1')
    except UnicodeEncodeError:
        return ''
    
# Replace problematic characters with suitable alternatives
            
            #urllib.urlretrieve(image, "magic_card_{count}.jpeg")

# Get the absolute path of the script
script_path = os.path.abspath(__file__)

# Get the directory containing the script
directory = os.path.dirname(script_path)

# Change the current working directory to the script's directory
os.chdir(directory)
place = os.getcwd()
print('this is place')
print(place)
data = json.load(open(f'{place}\\static\\cards_object.json'))
count = 0
for key in data:
    if(count == 0):
        print(data[key].values())
    count = count + 1
    # try:
    #     text_list = [value+'\n\n' for value in data[key]['text'].values()]
    #     combined_text = ' '.join(text_list)
    # except:
    #     combined_text = data[key]['text']
    combined_text = data[key]['text']
    color = "grey"   
    try:
        multi = 0
        for each in data[key]['colors']:
            multi = 0
            color = "grey"
            if each == 'W':
                color = "white"
                multi += 1
            if each == 'U':
                color = "blue"
                multi += 1
            if each == 'B':
                color = "black"
                multi += 1
            if each == 'G':
                color = "green"
                multi += 1
            if each == 'R':
                color = "red"
                multi += 1
        if(multi) > 2:
            color = "yellow"
        elif(multi>1):
            color = "yellow"
    except:
        pass
    if(combined_text is None):
        combined_text = "Vanilla"
    from textwrap import TextWrapper
    tw = TextWrapper()
    tw.width = 50
    wrappe = "\n".join(tw.wrap(combined_text))    
    Keep_old_images = False
    try:
        _return_image(color,key,combined_text,data[key]['image_url'],Keep_old_images)
    except:
        path= os.getcwd()+'\\static\\images\\logo.jpg'
        foreground = Image.open(path).convert("RGBA")
        path= os.getcwd()+'\\static\\images\\blank card.jpg'
        background = Image.open(path)
        background = write_image(background,colors['purple'],f'card_number_{count}','error',foreground=foreground)
        dir = os.getcwd()+'\\static\\images\\card_images'
        background.save(fr'{dir}\\error_card_number_{count}.jpeg')
print('completed')