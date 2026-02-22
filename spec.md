# Project: testing a CNN for use with images of fibrin clots

## project description
Train and test a CNN in python to classify images of fibrin clots between different phenotypes (normal plasma and several factor deficient plasmas).

### desired behavior for LLM assistants
Work as an expert data scientist to perform a practical assesment of the capability of CNNs to naively classify the fibrin images as specified. Refer to `spec.md` for project specifications but do not hesitate to ask for clarification or to challenge assumptions contained therein when needed.

### proposed project structure
Write code that will:
- load the image data easily in a pytorch-native format, perhaps into a pytorch native iterable
- transform image data as necessary into a greyscale format, perhaps using color spaces to capture the most information possible. write tests that show the human user the effects of these grayscale transformations.
- Perform data augmentation - at this point, keep it to vertically and horizontally flipping the rectangular images to take advantage of the four-fold symmetry. 
- Train a CNN with the majority (70-80%) of the data - using training images together with their augmentations and no other images - in a class-balanced manner. Also record the images used in training.
- Test the CNN's behavior with the remaining images and evaluate its accuracy in detail, including a breakdown by class.
- provide suggestions for project or architecture improvements without unnecessary scope drift

### proposed deep learning structure
- use square convolutional kernels and avoid changing the image aspect ratio to preserve rotational symmetry
- if needed to satisfy computational constraints, perform pooling operations (mean or min-pooling) that will preserve the unique dark fiber structure in the images before the convolutional operations
- ensure that features on the scale of at least 1/20 of the larger length of the image are identifiable by the use of kernel(s), in order to allow for detection of unique features at the junction of different fibers.
- Carefully document the desired CNN architecture in the code and in the markdown documentation so the user can evaluate it

### libraries to use: 
Work inside of the conda virtual environment `fibrin`, which I will activate if not already activated. See more detailed environment information in `environment.yml`.
- pytorch for cnns
- opencv, etc for image processing steps
- sqlalchemy for reading from the database
- pandas and numpy for general data processing steps
- pytest for writing and performing tests

### coding style:
Be efficient, both computationally and code-wise. Comment code with explanations of each step without being too verbose or lengthy. Also provide human-readable documentation in markdown format in a single file. Aim for readable, modest code that gets the job done without overcomplicating things. Store important methods and classes in respective `.py` files. Write tests in separate `test_<filename>.py` files for each critical project component so that a human user can easily run the tests and verify expected output. For example, a test that data augmentation is working as expected could show an example image pre-augmentation and display the three augmented versions in additional to the original image using a plotting library installed in the conda environment.

### images description
Images are stored in `./data/photos/`. The images are 6000 x 4000 pixels. The images are mostly a homogeneous light color with dark fibrous patches stretching across part or all of the images. The assumption is that the pattern, density, etc of the fibrous patches is correlated with the blood phenotype.

### labels description
Image labels are stored in `./data/test_db.db`. The labels are coded to the image filename. The images are named `\d\d\d\d.JPG` with a four digit (including padded zeros) code, which when converted to an int corresponds to the row of the image in the `test_db.db` database, starting with row zero.

### available computational resources
System76 Darter Pro
96.0 GiB
Intel® Core™ Ultra 7 255H × 16
Mesa Intel® Graphics (ARL)