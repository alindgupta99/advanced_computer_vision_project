To start with the training process, we did the below setup in our machine. Please follow this:

sudo apt-get update
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libglm-dev

pip install b2


function configure_b2() {
    b2 authorize-account $1 $2
}

configure_b2 c063d873e14e 004fb2500f70554bfa2d850f0f656f3c757f887af1

# b2 authorize-account c063d873e14e 004fb2500f70554bfa2d850f0f656f3c757f887af1

sudo apt-get install zip
sudo apt-get install unzip
