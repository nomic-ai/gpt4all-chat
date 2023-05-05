from setuptools import setup, find_packages
import os
import platform
import shutil

package_name = "gpt4all"

# Define the location of your prebuilt C library files
SRC_CLIB_DIRECtORY = "../../llmodel/"
SRC_CLIB_BUILD_DIRECTORY = "../../llmodel/build"

LIB_NAME = "llmodel"

DEST_CLIB_DIRECTORY = f"{package_name}/{LIB_NAME}_DO_NOT_MODIFY"
DEST_CLIB_BUILD_DIRECTORY = os.path.join(DEST_CLIB_DIRECTORY, "build")


system = platform.system()

def get_c_shared_lib_extension():
    
    if system == "Darwin":
        return "dylib"
    elif system == "Linux":
        return "so"
    elif system == "Windows":
        return "dll"
    else:
        raise Exception("Operating System not supported")
    

lib_ext = get_c_shared_lib_extension()


def copy_prebuilt_C_lib(src_dir, src_build_dir, dest_dir, dest_build_dir):
    files_copied = 0

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        for item in os.listdir(src_dir):
            # copy over header files to dest dir
            if item.endswith(".h"):
                s = os.path.join(src_dir, item)
                d = os.path.join(dest_dir, item)
                shutil.copy2(s, d)
                files_copied += 1
    

    if not os.path.exists(dest_build_dir):
        os.mkdir(dest_build_dir)
        for item in os.listdir(src_build_dir):
            # copy over shared library to dest build dir
            if item.endswith(lib_ext):
                s = os.path.join(src_build_dir, item)

                # Need to copy .dll right next to Cython extension for Windows
                if system == "Windows":
                    d = os.path.join(".", item)
                    shutil.copy2(s, d)
                else:
                    d = os.path.join(dest_build_dir, item)
                
                shutil.copy2(s, d)
                files_copied += 1
    
    return files_copied


# NOTE: You must provide correct path to the prebuilt llmodel C library. 
# Specifically, the llmodel.h and C shared library are needed.
copy_prebuilt_C_lib(SRC_CLIB_DIRECtORY,
                    SRC_CLIB_BUILD_DIRECTORY,
                    DEST_CLIB_DIRECTORY,
                    DEST_CLIB_BUILD_DIRECTORY)

setup(
    name=package_name,
    version="0.1.6",
    description="Python bindings for GPT4All",
    author="Richard Guo",
    author_email="richard@nomic.ai",
    url="https://pypi.org/project/gpt4all/",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=["pytest==7.3.1"],
    package_data={'llmodel': [f"{DEST_CLIB_DIRECTORY}/*", f"*.dll"]},
    include_package_data=True
)