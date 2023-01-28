import setuptools

setuptools.setup(
    name="babylog",
    version="0.0.1",
    author="Ahmad Roumie",
    author_email="ahmad@babylonai.dev",
    description="Babylog Library",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.24.1",
        "PyYAML==6.0",
        "opencv-python==4.7.0.68",
        "boto3==1.15.3",
        "protobuf==4.21.12",
        "zmq==0.0.0",
    ],
)
