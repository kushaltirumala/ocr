# ocr
nn.py — feed-forward ANN design testing
ocr.py — uses backpropagation and the ANN to detect 20px by 20px digits with 15 hidden nodes

data.csv/dataLabels.csv — training data

improvements: 
- has a few issues with detecting 5's correctly
- detecting more than digits (extending to characters)


To run the code with your own handrawn image: take an picture or screenshot of a digit, turn the image into 20pixels by 20 pixels, add it as the image (change <code>Image im = </code>), and run <code>python ocr.py</code>

