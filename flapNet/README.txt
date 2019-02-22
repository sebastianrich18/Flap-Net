Requires tensorflow, keras, kerboard, pygame, and numpy

Must be running python3

To install packages type "pip install [NAME]" into your IDE's console

Credit to Timo Wilken for making the game part: https://github.com/TimoWilken/flappy-bird-pygame

How to train the NN:
	1. twords the bottom of the code there is a commented out function named "writeData"
	   uncomment this function.
	2. set AI and the if statement containing the training code to False
	3. play the game, and data will be colected
	4. open 'dataCompresser.py' and run it once
	5. set the if statment containing the training code to True
	6. to test the NN, set the if statment containing the training code to False,
	   and set the other if statement to True, as well as set AI to true
