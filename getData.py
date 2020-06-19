import os
import zipfile
import easygui

def unZipAll(dir):
    for path in os.listdir(dir):
        s_path = os.path.splitext(path)
        if s_path[1].lower() == '.zip':
            local_zip = os.path.join(dir, path)
            zip_ref = zipfile.ZipFile(local_zip, 'r')
            new_dir = os.path.join(dir, s_path[0])
            zip_ref.extractall(new_dir)
            zip_ref.close()

# load images of humans and horses from ZIP-file
def getData():
    # add generalized directory
    train_dir = None
    while train_dir is None:
        train_dir = easygui.diropenbox(title="Choose folder that contains binary training set split in two folders" )
        if len(os.listdir(train_dir)) != 2:
            if easygui.ynbox("Directory must contain two sub-directories.\nChoose another directory?"):
                train_dir = None
            else:
                exit(1)


    # validation data is not mandatory
    validation_dir = easygui.diropenbox(title="Choose directory that contains validation data of two types" )

    # if directory is chosen - unzipping is not needed
    # if not unzipped:
    #     unZipAll(base_dir)

    # train_dir = os.path.join(base_dir, name_dir)
    # validation_dir = os.path.join(base_dir, 'validation-' + name_dir)

    train_dirs = [os.path.join(train_dir, x)
                  for x in os.listdir(train_dir)]

    validation_dirs = [None, None]
    if not validation_dir is None:
        validation_dirs = [os.path.join(validation_dir, x)
                           for x in os.listdir(validation_dir)]

    return tuple(train_dirs + validation_dirs)

# import tkinter as tk
# from tkinter import filedialog


def getUserFile(a_label: str, b_label: str):
# this does not work well - hangs the script
#     root = tk.Tk()
#     root.withdraw()
#     filename = filedialog.askopenfilename()
#     return filename

# easygui works better

     return easygui.fileopenbox(
          msg="{} or {}".format(a_label, b_label),
          title="Choose image(s)",
          multiple=True
     )

