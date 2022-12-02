import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d(dataset_item, threshold=(-0.75, 0.2)):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    face_color = ([0.5, 0.5, 1], [1, 0, 0])
    
    for i, key in enumerate(dataset_item.keys()):
        pic = dataset_item[key]['data'].squeeze().numpy().transpose(1, 0, 2)
        verts, faces, normals, values = measure.marching_cubes(pic, threshold[i], method='lewiner')    
        mesh = Poly3DCollection(verts[faces], alpha=0.2)
        mesh.set_facecolor(face_color[i])
        ax.add_collection3d(mesh)
        ax.set_xlim(0, pic.shape[0])
        ax.set_ylim(0, pic.shape[1])
        ax.set_zlim(0, pic.shape[2])

    plt.show()
    
def plot_slice(dataset_item, zslice=10):
    image = (dataset_item['image']['data'].squeeze()[:,:,zslice]).numpy()
    segmentation = (dataset_item['segmentation']['data'].squeeze()[:,:,zslice]).numpy()

    fig = plt.figure(figsize = (20, 15))
    rows = 1
    columns = 3
    overlay = color.label2rgb(segmentation, image, alpha=0.7, bg_label=0.9, bg_color=None, image_alpha=1, colors=[(0.6, 0.6, 0.6), (0.612, 0.324, 0.923)], kind = 'overlay')
    
    # Image
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image, cmap = 'gray', origin = 'lower')
    # plt.axis('off')
    plt.title("Image")
    
    # Segmentation
    fig.add_subplot(rows, columns, 2)
    plt.imshow(segmentation, cmap = 'gray', origin = 'lower')
    # plt.axis('off')
    plt.title("Segmentation")
    
    # Overlay
    fig.add_subplot(rows, columns, 3)
    plt.imshow(overlay, cmap = 'gray', origin = 'lower')
    # plt.axis('off')
    plt.title("Substract")
    
    plt.show()
    
