# gpt4all-chat

Cross platform Qt based GUI for GPT4All versions with GPT-J as the base
model. NOTE: The model seen in the screenshot is actually a preview of a
new training run for GPT4All based on GPT-J. The GPT4All project is busy
at work getting ready to release this model including installers for all
three major OS's. In the meantime, you can try this UI out with the original
GPT-J model by following build instructions below.

![image](https://user-images.githubusercontent.com/50458173/231464085-da9edff6-a593-410e-8f38-7513f75c8aab.png)

## Features

* Cross-platform (Linux, Windows, MacOSX)
* Fast CPU based inference using ggml for GPT-J based models
* The UI is made to look and feel like you've come to expect from a chatty gpt
* Check for updates so you can alway stay fresh with latest models
* Easy to install with precompiled binaries available for all three major desktop platforms
* Multi-modal - Ability to load more than one model and switch between them
* Supports both llama.cpp and gptj.cpp style models
* Model downloader in GUI featuring many popular open source models
* Settings dialog to change temp, top_p, top_k, threads, etc
* Copy your conversation to clipboard
* Check for updates to get the very latest GUI

## Feature wishlist

* Multi-chat - a list of current and past chats and the ability to save/delete/export and switch between
* Text to speech - have the AI response with voice
* Speech to text - give the prompt with your voice
* Python bindings
* Typescript bindings
* Plugin support for langchain other developer tools
* Save your prompt/responses to disk
* Upload prompt/respones manually/automatically to nomic.ai to aid future training runs
* Syntax highlighting support for programming languages, etc.
* REST API with a built-in webserver in the chat gui itself with a headless operation mode as well
* Advanced settings for changing temperature, topk, etc. (DONE)
* YOUR IDEA HERE

## Getting the latest

If you've already checked out the source code and/or built the program make sure when you do a git fetch to get the latest changes and that you also do ```git submodule update --init --recursive``` to update the submodules.

## Building and running

* Install Qt 6.x for your platform https://doc.qt.io/qt-6/get-and-install-qt.html
* Install cmake for your platform https://cmake.org/install/
* Download https://huggingface.co/EleutherAI/gpt-j-6b
* Clone this repo and build
```
git clone https://github.com/ggerganov/ggml.git
cd ggml
mkdir build
cd build
cmake ..
cmake --build . --parallel
python3 ../ggml/examples/gpt-j/convert-h5-to-ggml.py /path/to/your/local/copy/of/EleutherAI/gpt-j-6B 0
./bin/gpt-j-quantize /path/to/your/local/copy/of/EleutherAI/gpt-j-6B/ggml-model-f32.bin ./ggml-model-q4_0.bin 2
```
and then
```
git clone --recurse-submodules https://github.com/nomic-ai/gpt4all-chat
cd gpt4all-chat
mkdir build
cd build
cmake ..
cmake --build . --parallel
mv /path/to/ggml-model-q4_0.bin bin
./bin/chat
```

## To get Qt installed for your system

* Highly advise using the official Qt online open source installer.
* You can obtain this by creating an account on qt.io and downloading the installer. 
* You should get latest Qt {Qt 6.5.x} for your system and the developer tools including QtCreator, cmake, ninja.
* WINDOWS NOTE: you need to use the mingw64 toolchain and not msvc
* ALL PLATFORMS NOTE: the installer has options for lots of different targets which will add a lot
of download overhead. You can deselect webassembly target, android, sources, etc to save space on your disk.

## Manual download of models
* https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin (default) (md5sum 81a09a0ddf89690372fc296ff7f625af) Current best commercially licensable model based on GPT-J and trained by Nomic AI on the latest curated GPT4All dataset.
* https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin (md5sum 91f886b68fbce697e9a3cd501951e455) Current best non-commercially licensable model based on Llama 13b and trained by Nomic AI on the latest curated GPT4All dataset.
* https://gpt4all.io/models/ggml-gpt4all-j-v1.2-jazzy.bin (md5sum 879344aaa9d62fdccbda0be7a09e7976) An commercially licensable model based on GPT-J and trained by Nomic AI on the v2 GPT4All dataset.
* https://gpt4all.io/models/ggml-gpt4all-j-v1.1-breezy.bin (md5sum 61d48a82cb188cceb14ebb8082bfec37) An commercially licensable model based on GPT-J and trained by Nomic AI on the v1 GPT4All dataset.
* https://gpt4all.io/models/ggml-gpt4all-j.bin (md5sum 5b5a3f9b858d33b29b52b89692415595) An commercially licensable model based on GPT-J and trained by Nomic AI on the v0 GPT4All dataset.
* https://gpt4all.io/models/ggml-vicuna-7b-1.1-q4_2.bin (md5sum 29119f8fa11712704c6b22ac5ab792ea) An non-commercially licensable model based on Llama 7b and trained by teams from UC Berkeley, CMU, Stanford, MBZUAI, and UC San Diego.
* https://gpt4all.io/models/ggml-vicuna-13b-1.1-q4_2.bin (md5sum 95999b7b0699e2070af63bf5d34101a8) An non-commercially licensable model based on Llama 13b and trained by teams from UC Berkeley, CMU, Stanford, MBZUAI, and UC San Diego.
* https://gpt4all.io/models/ggml-wizardLM-7B.q4_2.bin (md5sum 99e6d129745a3f1fb1121abed747b05a) An non-commercially licensable model based on Llama 7b and trained by Microsoft and Peking University.
* https://gpt4all.io/models/ggml-stable-vicuna-13B.q4_2.bin (md5sum 6cb4ee297537c9133bddab9692879de0) An non-commercially licensable model based on Llama 13b and RLHF trained by Stable AI.

## Terminal Only Interface with no Qt dependency

Check out https://github.com/kuvaus/LlamaGPTJ-chat which is using the llmodel backend so it is compliant with our ecosystem and all models downloaded above should work with it.

## Contributing

* Pull requests welcome. See the feature wish list for ideas :)


## License
The source code of this chat interface is currently under a MIT license. The underlying GPT4All-j model is released under non-restrictive open-source Apache 2 License.

The GPT4All-J license allows for users to use generated outputs as they see fit. Users take responsibility for ensuring their content meets applicable requirements for publication in a given context or region.
