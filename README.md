# Welcome to Babylog
Welcome to babylog, a Python library, designed to stream image and video data from edge devices to the cloud with ease. Babylog is maintained by [BabylonAI, a Zurich-based YC-backed startup](https://babylonai.dev).

The primary goal of this library is to empower machine learning teams to log images and predictions, ensuring their computer vision models are working as intended. Without proper monitoring, small issues with a computer vision model can have dire consequences, such as a self-driving car incorrectly identifying a stop sign and causing an accident. Babylog aims to prevent such scenarios by providing the tools necessary for monitoring and debugging computer vision models.

# Installation & requirements
_Currently babylog only supports streaming the data to AWS. If you are using GCP or another provider please get in touch with us at founders@babylonai.dev and we'll make sure to add it into the development pipeline._

## Supported python versions
The babylog Python library is compatible with **Python version 3.7 and above**. It is recommended to use the latest version of Python for best performance and stability. If you are using an older version of Python, you may need to upgrade your Python installation in order to use babylog. You can check your Python version by running the command `python --version` in your command prompt or terminal.

## Installation
Like most python packages, run: 

```bash
pip3 install babylog
```

## Getting and configuring your AWS credentials
- Open the IAM console at https://console.aws.amazon.com/iam/
- On the navigation menu, choose Users.
- Choose your IAM user name (not the check box).
- Open the Security credentials tab, and then choose Create access key.
- To see the new access key, choose Show. Your credentials resemble the following:
  - Access key ID: AKIAIOSFODNN7EXAMPLE
  - Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
