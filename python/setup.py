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
    install_requires=["numpy", "PyYAML", "opencv-python", "boto3", "protobuf"],
)
