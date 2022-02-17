# img_viewer.py

import PySimpleGUI as sg
import os.path
import converter


beingConverted = False
filename = ''
opname=''

# First the window layout in 2 columns
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [
        sg.Text("Save File As:"),
        sg.In(size=(25,1),enable_events=True,key='-NAMED-')
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text(key="-MESSAGE-")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.Button('Convert'),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Medi Modellor", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == 'Convert':
        beingConverted = True
        window["-MESSAGE-"].update('Converting Img to 3D model...')
        converter.imageProcessing(filename,opname)
        print('Done')
        beingConverted = False
        window["-MESSAGE-"].update('Choose an image from list on the left')
    if event == "-FOLDER-":
        if beingConverted == True:
            window["-MESSAGE-"].update('Converting Img to 3D model...')
            pass
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif","jpg","jpeg"))
        ]
        window["-FILE LIST-"].update(fnames)
    if event == '-NAMED-':
        opname=values['-NAMED-']
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        if beingConverted == True:
            window["-MESSAGE-"].update('Converting Img to 3D model...')
            pass
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)
        except:
            pass

window.close()
