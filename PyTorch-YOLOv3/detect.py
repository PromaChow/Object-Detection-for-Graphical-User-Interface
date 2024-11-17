from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import csv
from PIL import Image
import glob

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class RGBImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.[pP][nN][gG]" % folder_path) + 
                          glob.glob("%s/*.*[jJ][pP][eE]*[gG]" % folder_path))
        self.img_size = img_size
        print(f"\nRGBImageFolder initialized with {len(self.files)} images")
        for f in self.files:
            print(f"- {f}")

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        
        # Extract image
        img = Image.open(img_path)
        
        # Convert RGBA to RGB if necessary
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Convert to RGB
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            elif img.mode == 'LA':
                background.paste(img, mask=img.split()[1])  # Use alpha channel as mask
            elif img.mode == 'P':
                background.paste(img, mask=img.getchannel('A'))  # Use transparency as mask
            img = background

        # Resize
        img = img.resize((self.img_size, self.img_size))
        
        # Convert to RGB (if not already)
        img = img.convert('RGB')
        
        # Convert to numpy, transpose to channels first
        img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = img / 255.0
        img = torch.from_numpy(img).float()

        return img_path, img

    def __len__(self):
        return len(self.files)

    def _pad_to_square(self, img):
        h, w = img.shape[:2]
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        img = np.pad(img, pad, 'constant', constant_values=127.5)
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--dataset", type=str, default="rico", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--csv_path", type=str)
    opt = parser.parse_args()
    
    # Debug: Print all arguments
    print("\nArguments:")
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    opt.model_def = "config/yolov3-{}.cfg".format(opt.dataset)
    opt.class_path = "data/{}/classes.names".format(opt.dataset)
    

    # Check if image folder exists and list contents
    print("\nChecking image folder:")
    if not os.path.exists(opt.image_folder):
        print(f"Error: Image folder '{opt.image_folder}' does not exist!")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)

    # Set up model
    try:
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        print("\nModel initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    try:
        if opt.weights_path.endswith(".weights"):
            model.load_darknet_weights(opt.weights_path)
            print("Loaded darknet weights successfully")
        else:
            model.load_state_dict(torch.load(opt.weights_path))
            print("Loaded checkpoint weights successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    model.eval()  # Set in evaluation mode

    # Initialize dataloader with our custom RGBImageFolder
    try:
        dataloader = DataLoader(
            RGBImageFolder(opt.image_folder, img_size=opt.img_size),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )
        print("\nDataloader initialized successfully")
    except Exception as e:
        print(f"Error initializing dataloader: {e}")
        sys.exit(1)

    try:
        classes = load_classes(opt.class_path)  # Extracts class labels from file
        print(f"\nLoaded {len(classes)} classes: {classes}")
    except Exception as e:
        print(f"Error loading classes: {e}")
        sys.exit(1)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    # Prepare CSV file for saving results
    csv_file = open(f'output/{opt.csv_path}', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_name', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'width', 'height'])

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        print(f"\nProcessing batch {batch_i + 1}")
        print(f"Image paths in batch: {img_paths}")
        print(f"Input tensor shape: {input_imgs.shape}")
        
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            try:
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                print(f"Detections in batch: {len([d for d in detections if d is not None])} objects")
            except Exception as e:
                print(f"Error during detection: {e}")
                continue

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images and detection results:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        try:
            print(f"\nProcessing image {img_i + 1}/{len(imgs)}")
            print(f"Image path: {path}")
            
            filename = path.split("/")[-1].split(".")[0]
            
            # Create plot
            img = Image.open(path)
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                elif img.mode == 'LA':
                    background.paste(img, mask=img.split()[1])
                elif img.mode == 'P':
                    background.paste(img, mask=img.getchannel('A'))
                img = background
            img = img.convert('RGB')
            img = np.array(img)
            print(f"Loaded image shape: {img.shape}")
            
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                print(f"Number of detections: {len(detections)}")
                
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print(f"Detection: {classes[int(cls_pred)]} (conf: {cls_conf.item():.3f})")
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    # Save detection results to CSV
                    csv_writer.writerow([
                        filename,
                        classes[int(cls_pred)],
                        cls_conf.item(),
                        x1.item(),
                        y1.item(),
                        x2.item(),
                        y2.item(),
                        box_w.item(),
                        box_h.item()
                    ])

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    ax.add_patch(bbox)

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            output_path = f"output/images/{filename}.png"
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()
            print(f"Saved annotated image to: {output_path}")
            
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            continue

    # Close CSV file
    csv_file.close()
    print("\nDetection results saved to output/detection_results.csv")
    print("Annotated images saved to output/images/")