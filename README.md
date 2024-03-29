# Welcome to Babylog

[![PyPI version](https://badge.fury.io/py/babylog.svg)](https://badge.fury.io/py/babylog)
[![pypi supported versions](https://img.shields.io/pypi/pyversions/babylog.svg)](https://pypi.python.org/pypi/babylog)
[![GitHub Super-Linter](https://github.com/thebabylonai/babylog/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

Welcome to babylog, a Python library, designed to stream image and video data from edge devices to the cloud with ease. Babylog is maintained by [BabylonAI, a Zurich-based YC-backed startup](https://babylonai.dev).

Check out the documentation here: https://babylonai.gitbook.io/babylog/

![github_background](https://user-images.githubusercontent.com/16129326/215295136-a10e6cf7-916a-47a3-a40e-6565a10d4a06.png)

The primary goal of this library is to empower machine learning teams to log images and predictions, ensuring their computer vision models are working as intended. Without proper monitoring, small issues with a computer vision model can have dire consequences, such as a self-driving car incorrectly identifying a stop sign and causing an accident. Babylog aims to prevent such scenarios by providing the tools necessary for monitoring and debugging computer vision models.

# Installation & requirements
_Currently babylog only supports streaming the data to AWS. If you are using GCP or another provider please get in touch with us at founders@babylonai.dev and we'll make sure to add it into the development pipeline._

## Supported python versions
The babylog Python library is compatible with **Python version 3.8 and above**. It is recommended to use the latest version of Python for best performance and stability. If you are using an older version of Python, you may need to upgrade your Python installation in order to use babylog. You can check your Python version by running the command `python --version` in your command prompt or terminal.

## Installation
Like most python packages, run: 

```bash
pip install babylog
```

## Getting and configuring your AWS credentials
- Open the IAM console at https://console.aws.amazon.com/iam/
- On the navigation menu, choose Users.
- Choose your IAM user name (not the check box).
- Open the Security credentials tab, and then choose Create access key.
- To see the new access key, choose Show. Your credentials resemble the following:
  - Access key ID: AKIAIOSFODNN7EXAMPLE
  - Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

### Get your bucket name and region
You also need the name of your bucket and a slug of the region. Either create a new bucket or find the values for an existing bucket in your S3 console: https://s3.console.aws.amazon.com/s3/buckets

<img width="1076" alt="Screenshot 2023-01-28 at 12 42 30" src="https://user-images.githubusercontent.com/16129326/215273827-d2260884-4570-4ca2-b46e-9b2c1ca21583.png">

### Creating a config file
You can set your AWS credentials in a `babylog.config.yaml` file to use them with the babylog library. Here is an example of how you might structure the file:

```yaml
device:
  ip: 'DEVICE_IP'       # device ip to be used for tcp streaming (e.g. '127.0.0.1')
  port: DEVICE_PORT     # port number(int) for tcp streaming
  name: 'DEVICE_NAME'   # device name (e.g. 'device-1b2a')
  group: 'GROUP_NAME'   # group name (e.g. 'group-222X)'
data:
  interval: 3000        # ms between captures
  max_workers: 4        # max number of threads to be used for logging
S3_storage:
  aws_access_key_id: 'YOUR_ACCESS_KEY'
  aws_secret_access_key: 'YOUR_SECRET_KEY'
  bucket_name: 'YOUR_BUCKET_NAME'
  bucket_region: 'YOUR_BUCKET_REGION'
```

You can replace YOUR_ACCESS_KEY and YOUR_SECRET_KEY with the actual values of your credentials. Currently you need one file per streaming device. If you want to add more granularity, please consider opening a github issue.

**We recommend keeping the config files locally for security purposes. Consider adding the file to your .gitignore.**
