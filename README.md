# Energydl: A tool for energy-aware interactive training of neural network architectures

This repository contains the code implementation corresponding to the following research paper:

> Álvaro Domingo, Silverio Martínez-Fernández, Roberto Verdechia. Energy-aware training of neural network architectures: Tradeoff between correctness and energy consumption.

 The code demonstrates the algorithms and methods outlined in the paper, providing a practical reference for researchers and developers interested in replicating the results or exploring the proposed techniques further.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/alvarodr21/energydl.git
   ```

2. Navigate to the project directory:

   ```
   cd energydl
   ```

3. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   ```

4. Activate the virtual environment:

   On Windows:

   ```
   venv\Scripts\activate
   ```

   On macOS and Linux:

   ```
   source venv/bin/activate
   ```

5. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

The module is contained in the file `energydl.py`. It can be imported to use in any Python script by `from energydl import energy_aware_train`. This function is aimed to replace TensorFlow's `tf.keras.Model.fit` on any script, passing the model as the first argument and keeping the rest. The behavior of it can also be customized by some unique parameters not present in `Model.fit`. 

A demo of a simple usage of this function is included in `demo_energydl.py`. It can be simply called with `python demo_energydl.py`.

