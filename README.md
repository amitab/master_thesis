# How to Run:

From the `docker` folder, run this command:
`docker-compose up -d --build --remove-orphans`

Then navigate to `localhost:10000` to access the jupyter notebook.

# The project structure and purpose

Each folder in the `src` directory is for a different ML task.
The purpose of this repository is for me to experiment with different ML tasks using one docker image having all dependencies installed.

# Where to get the data

## FF data:

https://www.kaggle.com/c/titanic/data
OR
kaggle competitions download -c titanic

## mnist data:

https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer/data


## mnist-fashion data:

https://www.kaggle.com/shivamb/how-autoencoders-work-intro-and-usecases/data


## storage_testing data:

Downloads the data when you run the script


## fisher data:

Clone: `git@github.com:learning1234embed/NeuralWeightVirtualization.git`

Run: `./download_dataset.sh && python weight_virtualization.py -mode=a -network_path=mnist`

Copy the contents of the mnist folder except the `.py(c)` files