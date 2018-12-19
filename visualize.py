import argparse
import cv2
import davis
import os

from scipy.misc import imsave

DATABASE_DIR = "/Users/khoipham/Documents/umd/research/workspace/databases/davis-2017/data/DAVIS"
ALPHA = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize results")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="directory to write results")
    parser.add_argument("--ov_result_dir", type=str, default="overlay_results",
                        help="directory to save overlayed images")
    parser.add_argument("--fname", type=str, default="None",
                        help="video name that you want to visualize")
    args = parser.parse_args()

    folders = os.listdir(args.result_dir)
    for fname in folders:
        if args.fname != "None" and fname != args.fname:
            continue
        print("Processing %s" % fname)
        folder_dir = os.path.join(DATABASE_DIR, "JPEGImages", "480p", fname)
        img_list = sorted(os.listdir(folder_dir))

        for i in range(len(img_list)):
            img_name = img_list[i].split(".")[0]
            if len(img_name) == 0:
                continue
            img = cv2.imread(os.path.join(folder_dir, img_list[i]))
            # mask = cv2.imread(os.path.join(args.result_dir, fname,
            #                                img_name + ".png"), 0)
            mask = davis.io.imread_indexed(
                os.path.join(args.result_dir, fname, img_name + ".png"))[0]

            res = img.copy()
            mask_pos = mask > 0
            res[mask_pos] = [0, 0, 255*ALPHA] + (1 - ALPHA) * res[mask_pos]

            output_folder = os.path.join(args.ov_result_dir, fname)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cv2.imwrite(os.path.join(output_folder, img_list[i]), res)
