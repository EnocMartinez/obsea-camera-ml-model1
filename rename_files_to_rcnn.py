# import pandas as pd
import os
from rich import print as rprint
# import xml.etree.cElementTree as ET
# from tqdm.auto import tqdm
import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import shutil


# Comprovem si tenim el directory i les subcarpetes bé
rprint('[yellow]Directori actual:', os.getcwd())
print()
rprint('[yellow]Fitxers i Carpetas dins del Directori:', os.listdir())
print('\n')

folder_data_XML_name = './Tagging_XML/'
if not os.path.exists(folder_data_XML_name):
    os.mkdir(folder_data_XML_name)
else:
    pass
xml_list_names = sorted(os.listdir(os.path.join(folder_data_XML_name)))
rprint(xml_list_names, '\n')

folder_data_Img_name = './Tagging_Img/'
if not os.path.exists(folder_data_Img_name):
    os.mkdir(folder_data_Img_name)
else:
    pass
img_list_names = sorted(os.listdir(os.path.join(folder_data_Img_name)))
rprint(img_list_names, '\n')

for i in range(0, len(xml_list_names)):
    if xml_list_names[i][:-4] != img_list_names[i][:-4]:
        print(xml_list_names[i][:-4], img_list_names[i][:-4])
        print('No estan ordenats igual')
    else:
        pass

new_names_list = []

for i in range(1, len(xml_list_names)+1):
    str = f'{i:05}'
    new_names_list.append(str)

print(new_names_list)
#
#
# for i in range(0, len(new_names_list)):
#
#     file_oldname_xml = os.path.join(folder_data_XML_name, xml_list_names[i])
#     print(file_oldname_xml)
#     file_newname_newfile_xml = os.path.join(folder_data_XML_name, f'{new_names_list[i]}.xml')
#     print(file_newname_newfile_xml)
#     os.rename(file_oldname_xml, file_newname_newfile_xml)
#
#     file_oldname_img = os.path.join(folder_data_Img_name, img_list_names[i])
#     print(file_oldname_img)
#     file_newname_newfile_img = os.path.join(folder_data_Img_name, f'{new_names_list[i]}.jpg')
#     print(file_newname_newfile_img)
#     os.rename(file_oldname_img, file_newname_newfile_img)
#
