import cv2
from PIL import Image
import pathlib
import numpy as np

if __name__ == '__main__':
    dataset_root_path = "gifs/DL project - gifs"
    dataset_root_path = pathlib.Path(dataset_root_path)
    dataset_target_path = "gifs/DL project - gifs - avi"
    dataset_target_path = pathlib.Path(dataset_target_path)
    video_file_paths_class_g = list(dataset_root_path.glob("test/g/*.gif"))
    video_file_paths_class_pg = list(dataset_root_path.glob("test/pg/*.gif"))
    video_file_paths_class_pg13 = list(dataset_root_path.glob("test/pg-13/*.gif"))
    video_file_paths_class_r = list(dataset_root_path.glob("test/r/*.gif"))

    for gif_path in (video_file_paths_class_g + video_file_paths_class_pg13 + video_file_paths_class_pg + video_file_paths_class_r):
        gif = Image.open(gif_path)

        split_path = gif_path.parts
        avi_path = "gifs/DL project - gifs/test_avi/" + split_path[-2] + "/" + split_path[-1] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        try:
            duration = gif.info['duration']
            fps = 1000 / duration
        except Exception:
            fps = 24

        width, height = gif.size

        # Create VideoWriter object
        out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

        # Loop over all frames of the GIF and write to AVI
        try:
            while True:
                # Convert each GIF frame (PIL Image) to a NumPy array for OpenCV
                frame = np.array(gif.convert('RGB'))  # Convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

                # Write the frame to the AVI video file
                out.write(frame)

                # Move to the next frame
                gif.seek(gif.tell() + 1)

        except EOFError:
            # End of GIF
            pass

        # Release the video writer
        out.release()

        print(f"Conversion complete! Saved as {avi_path}")
