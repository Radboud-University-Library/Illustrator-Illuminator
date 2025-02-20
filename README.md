# Illustrator Illuminator
<img src="Illustrator-Illuminator.png" alt="Illustrator Illuminator" width="330"/>

Image Classification and Sorting for Heritage Book Scans

## Overview
This project is designed to classify and sort images into different categories 
using a pre-trained [Teachable Machine](https://teachablemachine.withgoogle.com/) model. 
The images are processed in batches, and based on the model's predictions, they are moved to appropriate directories.

## Requirements
- Python 3.11
- TensorFlow
- Pillow
- NumPy
- Tkinter

## Installation
1. Install the required Python packages:
```bash
    pip install -r requirements.txt
```

## Usage
1. **Load the Model**: The model is loaded from a specified path.
2. **Select Directory**: A user interface is provided to select the directory containing images.
3. **Process Images**: Images in the selected directory are processed in batches. There are 5 categories:

- Illustration
- Text
- EmptyPages
- TitlePages
- Tables

4. **Move Images**: Based on the model's predictions, images are moved to the appropriate directories. 

## Training the Model
The model was trained using the [Teachable Machine](https://teachablemachine.withgoogle.com/) platform. 
The images in the training data was gathered from material from the Radboud University Library.
In the training data set, images were divided into 5 categories: Illustration, Text, EmptyPages, TitlePages, and Tables.
The number of image samples for each category was as follows:

- Illustration: 8019 Image Samples
- Text: 3710 Image Samples
- Empty Pages: 6832 Image Samples
- Title Pages: 905 Image Samples
- Image Samples: 2841 Image Samples

Why the differences in numbers? This is the last result of my experimenting with the model. Previous versions contained
different ratios and had different results.

## Testset

You can use the testset to test the model. The testset is located in the `testset` directory. It contains 60 images
(20 illustrations, 10 text, 10 empty pages, 10 title pages, and 10 tables). You'll see that its not perfect,
but it does a decent job separating images. You can use this script in combination with tools like Faststone Image Viewer
or IrfranView to quickly sort through the images. With this script Ive managed to extract +/- 25000 images from our 
digitized collection.


## Running the Project
To run the project, execute the `main.py` file:
```bash
  python main.py
```
    
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```