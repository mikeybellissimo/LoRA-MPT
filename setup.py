from setuptools import setup, find_packages

setup(
    name='mpt-lora',
    version='0.1.0',    
    description='A library for finetuning MPT',
    url='https://github.com/mikeybellissimo/LoRA-MPT',
    author='Michael Bellissimo',
    author_email='leucha@gmail.com',
    license='Apache 2.0',
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    install_requires=["sentencepiece", "ipykernel", "accelerate", "appdirs", "loralib", "bitsandbytes", "black", "black[jupyter]", "datasets", 
        "fire", "peft", "transformers>=4.28.0", "sentencepiece", "gradio", "einops"
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: Apache 2.0',  
        'Programming Language :: Python :: 3',
    ],
)