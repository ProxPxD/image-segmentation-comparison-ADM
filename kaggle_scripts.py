import os
import shutil

def move_and_remove_subdirectories(src_directory, dest_directory):
    for filename in os.listdir(src_directory):
        src_file = os.path.join(src_directory, filename)
        dest_file = os.path.join(dest_directory, filename)

        shutil.move(src_file, dest_file)

    os.rmdir(src_directory)
