# CAL Hacks

# What is this?
ASL-Quick-Learn is a website that allows a user to signal different ASL letters and gain feedback based on what they signal. The website generates a random letter, and the user signs that and takes a picture using a handy one-click button on the website. Immediately after they take the picture, they receive a point if their sign was correct, and the website generates a new random letter for them to signal. At anytime the user can see their current score at the bottom of the screen.


# Our Process
We had an extremely well-thought out step-by-step process for our code. We started by designing a webscraping script using Python and the library BeautifulSoup. This gathered images for each sign corresponding to a letter from an online ASL dictionary. Although currently, we only have support for ASL letters, in the future we want to add words and this script lets us scale fast. Next, after gathering the data we pre-processed the dataset by utilizing Google's API, Mediapipe, to help us label 21 distinct locations (each location contains an x, y, z value) on the hand that each distinct ASL gesture creates. From there we labeled each image with the corresponding letter and 63 values that are generated from each movement to make up the composition of the model's data. Designing a basic machine learning model, we were able to predict the letter corresponding a sign to a given image. This was done using technologies including TensorFlow, Mediapipe and OpenCV. Finally we got to the web
