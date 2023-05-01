import cv2
import numpy as np
import os
from os.path import isfile, join
import argparse


def convert_frames_to_video(pathIn,pathOut):
    images_dir = pathIn

    # Get a list of all the image files in the directory
    images = [img for img in os.listdir(images_dir) if img.endswith('.jpg')]

    # Sort the images based on their names
    images.sort(key = lambda x: int(x[3:-4]))

    # Read the first image to get the frame width and height
    frame = cv2.imread(os.path.join(images_dir, images[0]))
    frame_height, frame_width, _ = frame.shape

    # Define the output video file name, codec, frame rate and frame size
    out_file = pathOut
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25.0
    frame_size = (frame_width, frame_height)

    # Create a video writer object
    out = cv2.VideoWriter(out_file, fourcc, fps, frame_size)

    # Iterate over each image in the directory and write it to the output video file
    for image in images:
        image_path = os.path.join(images_dir, image)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the resources
    out.release()
def main(pathIn,pathOut):
    convert_frames_to_video(pathIn, pathOut)
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create video from images frames.")
    parser.add_argument("--pathIn",
                        help="Relative location of frames folder.",
                        default="./MVI_39031/")
    parser.add_argument("--pathOut",
                        help="Name of video.",
                        default="MVI_39031.mp4")
    args = parser.parse_args()
    pathIn=args.pathIn
    pathOut=args.pathOut
    main(pathIn,pathOut)