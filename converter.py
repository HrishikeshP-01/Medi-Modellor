from PIL import Image
import numpy as np
from stl import mesh
import cv2

def convertImage(imgPath,opname):
    grey_img = Image.open(imgPath).convert('L')

    max_size=(500,500)
    max_height=50
    min_height=-50

    #height=0 for minPix
    #height=maxHeight for maxPIx

    grey_img.thumbnail(max_size)
    imageNp = np.array(grey_img)
    maxPix=imageNp.max()
    minPix=imageNp.min()



    print(imageNp)
    (ncols,nrows)=grey_img.size

    vertices=np.zeros((nrows,ncols,3))

    for x in range(0, ncols):
        for y in range(0, nrows):
            pixelIntensity = imageNp[y][x]
            z = (pixelIntensity * max_height) / maxPix
            #print(imageNp[y][x])
            vertices[y][x]=(x, y, z)

    faces=[]

    for x in range(0, ncols - 1):
        for y in range(0, nrows - 1):
            # create face 1
            vertice1 = vertices[y][x]
            vertice2 = vertices[y+1][x]
            vertice3 = vertices[y+1][x+1]
            face1 = np.array([vertice1,vertice2,vertice3])

            # create face 2 
            vertice1 = vertices[y][x]
            vertice2 = vertices[y][x+1]
            vertice3 = vertices[y+1][x+1]

            face2 = np.array([vertice1,vertice2,vertice3])

            faces.append(face1)
            faces.append(face2)

    print(f"number of faces: {len(faces)}")
    facesNp = np.array(faces)
    # Create the mesh
    surface = mesh.Mesh(np.zeros(facesNp.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            surface.vectors[i][j] = facesNp[i][j]
    # Write the mesh to file "cube.stl"
    opname = opname+'.stl'
    surface.save(opname)
    print(surface)

def imageProcessing(imgPath,opname):
    img = cv2.imread(imgPath)
    #converted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    cv2.imwrite('processedimg.png',dst)
    convertImage('processedimg.png',opname)
