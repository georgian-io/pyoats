# Installation
## PyPI
1. Install package via pip
   ```sh
   pip install pyoats
   ```
  
## Docker
1. Clone the repo
    ```sh
    git clone https://github.com/georgian-io/pyoats.git && cd pyoats 
    ```
2. Build image
    ```sh
    docker build -t pyoats . 
    ```
3. Run Container
    ```sh 
    # CPU Only
    docker run -it pyoats
    
    # with GPU
    docker run -it --gpus all pyoats
    ```
    
## Local
1. Clone the repo
    ```sh
    git clone https://github.com/georgian-io/pyoats.git && cd pyoats 
    ```
2. Install via Poetry
    ```sh 
    poetry install
    ```

