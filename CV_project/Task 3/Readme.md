Certainly! Here is your README file formatted for better readability:

```markdown
# Setup Instructions for Training Process

To start with the training process, we did the below setup in our machine. Please follow these steps:

## System Update and Dependencies Installation

First, update your package list and install the necessary dependencies:

```sh
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libglm-dev
```

## Install B2

Install the B2 package using pip:

```sh
pip install b2
```

## Configure B2

Create a function to configure B2 with the provided account details:

```sh
function configure_b2() {
    b2 authorize-account $1 $2
}

configure_b2 c063d873e14e 004fb2500f70554bfa2d850f0f656f3c757f887af1
```

Alternatively, you can directly authorize the B2 account:

```sh
b2 authorize-account c063d873e14e 004fb2500f70554bfa2d850f0f656f3c757f887af1
```

## Install Zip and Unzip

Finally, install the zip and unzip utilities:

```sh
sudo apt-get install zip
sudo apt-get install unzip
```

Following these steps will set up your machine for the training process.
```

This structure should make the instructions clear and easy to follow.
