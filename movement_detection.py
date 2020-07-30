import numpy as np
import argparse
import cv2
import simpleaudio
import pygame
from datetime import datetime

from scipy.linalg import norm
from scipy import sum

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return np.average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def main(args):
    # Prepare audo playback objet
    pygame.mixer.init()
    pygame.mixer.music.load("./audio/owl.wav")

    # define a video capture object 
    vid = cv2.VideoCapture(0)
    prev_image = None
    detected = False
    n_frames = 0
    while(True):
        n_frames += 1
        # Capture the video frame 
        # by frame 
        _, image = vid.read()
        image = to_grayscale(image.astype(float))
        if prev_image is not None:
            n_m, n_0 = compare_images(image, prev_image)
            n_m_per_pix = n_m/image.size
            # n_0_per_pix = n_0*1.0/image.size

            prev_image = image

            current_time = datetime.now().strftime("%H:%M:%S")

            if n_m_per_pix > args.threshold:
                if n_frames > 20:
                    if detected:
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                            continue
                        detected = False
                    else:
                        detected = True 
                print(f"[INFO] {current_time} - Object detected: {n_m_per_pix}")

            else:
                print(f"[INFO] {current_time}")
        else:
            prev_image = image
    # After the loop release the cap object 
    vid.release()

def parse_arguments():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--threshold", type=float, default=2.5 ,
        help="minimum threshold for norm between frames")
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
