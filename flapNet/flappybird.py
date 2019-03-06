# modified game code from https://github.com/TimoWilken/flappy-bird-pygame

import os
from random import randint
from collections import deque
import pygame
from pygame.locals import *
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import keyboard as kb




AI = True # true if you want ai to play the game, false to disable ai
train = False # true if you want to train the NN, false if you do not want to train
game = True # true if you want the game to run, false if you dont want the game to run
collectData = '' # 'user' to write to data.txt from user, 'ai' if you want to write data from ai, '' or None if you do not want to write data




model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(3,)))
model.add(Dense(1, activation='sigmoid', input_shape=(8,)))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.load_weights('flapNetWeights.h5') # comment out if you do not want to load weights ---------------------------------------------------------------------------------------------

if train:
    
    allData = np.loadtxt("data.txt", delimiter=',')

    tempData = np.split(allData, 10)
    testData = tempData[0]
    testData = np.append(testData, tempData[1], axis=0)  # these lines split the data into 70% train data and 30% test data
    testData = np.append(testData, tempData[2], axis=0)
    trainData = tempData[3]
    for i in range(6):
        trainData = np.append(trainData, tempData[i+4], axis=0)
        
    
    
    history = model.fit(trainData[:,[0,1,2]], trainData[:,3],                                       #used to train model
                            batch_size=32,
                            epochs=10000, # change this to train longer
                            verbose=1,
                            validation_data=((testData[:,[0,1,2]], testData[:,3])))
                            
    model.save_weights("flapNetWeights.h5")   # comment out if you do not want to save weights **will replace old weights file**



if game: # set to True if you want the game to run
             
    FPS = 60
    ANIMATION_SPEED = 0.18  # pixels per millisecond
    WIN_WIDTH = 284 * 2     # BG image size: 284x512 px; tiled twice
    WIN_HEIGHT = 512
    loopCount = 0.
        
    class Bird(pygame.sprite.Sprite):
        WIDTH = HEIGHT = 32
        SINK_SPEED = 0.25
        CLIMB_SPEED = 0.3
        CLIMB_DURATION = 333.3
    
        def __init__(self, x, y, msec_to_climb, images):
            super(Bird, self).__init__()
            self.x, self.y = x, y
            self.yVel = 0.
            self.msec_to_climb = msec_to_climb
            self._img_wingup, self._img_wingdown = images
            self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
            self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)
    
        def getY(self):
            return round(self.y + 1.,7) 
            
        def getyVel(self):
            return round(self.yVel, 4)
    
        def update(self):
            self.y += self.yVel
            self.yVel += .4
    
        @property
        def image(self):
            if pygame.time.get_ticks() % 500 >= 250:
                return self._img_wingup
            else:
                return self._img_wingdown
    
        @property
        def mask(self):
            if pygame.time.get_ticks() % 500 >= 250:
                return self._mask_wingup
            else:
                return self._mask_wingdown
    
        @property
        def rect(self):
            return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)
    
    
    class PipePair(pygame.sprite.Sprite):    
        WIDTH = 80
        PIECE_HEIGHT = 32
        ADD_INTERVAL = 3000
    
        def __init__(self, pipe_end_img, pipe_body_img):
            self.x = float(WIN_WIDTH - 1)
            self.score_counted = False
            self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
            self.image.convert()   # speeds up blitting
            self.image.fill((0, 0, 0, 0))
            total_pipe_body_pieces = int(
                (WIN_HEIGHT -                  # fill window from top to bottom
                4 * Bird.HEIGHT -             # make room for bird to fit through
                3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
                PipePair.PIECE_HEIGHT          # to get number of pipe pieces
            )
            self.bottom_pieces = randint(1, total_pipe_body_pieces)
            self.top_pieces = total_pipe_body_pieces - self.bottom_pieces
    
            # bottom pipe
            for i in range(1, self.bottom_pieces + 1):
                piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
                self.image.blit(pipe_body_img, piece_pos)
            bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
            bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_end_img, bottom_end_piece_pos)
    
            # top pipe
            for i in range(self.top_pieces):
                self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
            top_pipe_end_y = self.top_height_px
            self.image.blit(pipe_end_img, (0, top_pipe_end_y))
    
            # compensate for added end pieces
            self.top_pieces += 1
            self.bottom_pieces += 1
    
            # for collision detection
            self.mask = pygame.mask.from_surface(self.image)
        
        def getX(self):
            return self.x
            
        @property
        def top_height_px(self):
            """Get the top pipe's height, in pixels."""
            return self.top_pieces * PipePair.PIECE_HEIGHT
    
        @property
        def bottom_height_px(self):
            """Get the bottom pipe's height, in pixels."""
            return self.bottom_pieces * PipePair.PIECE_HEIGHT
    
        @property
        def visible(self):
            """Get whether this PipePair on screen, visible to the player."""
            return -PipePair.WIDTH < self.x < WIN_WIDTH
    
        @property
        def rect(self):
            """Get the Rect which contains this PipePair."""
            return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)
    
        def getTopBottomY(self):
            return WIN_HEIGHT - self.bottom_height_px
        
        def update(self, delta_frames=1):
            """Update the PipePair's position.
    
            Arguments:
            delta_frames: The number of frames elapsed since this method was
                last called.
            """
            
            self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)
            
        def getXdist(self):
            return self.x - 80
    
        def collides_with(self, bird):
            """Get whether the bird collides with a pipe in this PipePair.
    
            Arguments:
            bird: The Bird which should be tested for collision with this
                PipePair.
            """
            return pygame.sprite.collide_mask(self, bird)
    
    
    def load_images():

    
        def load_image(img_file_name):
            file_name = os.path.join('.', 'images', img_file_name)
            img = pygame.image.load(file_name)
            img.convert()
            return img
    
        return {'background': load_image('background.png'),
                'pipe-end': load_image('pipe_end.png'),
                'pipe-body': load_image('pipe_body.png'),
                # images for animating the flapping bird -- animated GIFs are
                # not supported in pygame
                'bird-wingup': load_image('bird_wing_up.png'),
                'bird-wingdown': load_image('bird_wing_down.png')}
    
    
    def frames_to_msec(frames, fps=FPS):
        return 1000.0 * frames / fps
    
    
    def msec_to_frames(milliseconds, fps=FPS):
        return fps * milliseconds / 1000.0
    
    
    
    
    def main():
        pygame.init()
        
        display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Pygame Flappy Bird')
        
        pygame.display.flip()
        
        clock = pygame.time.Clock()
        score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
        images = load_images()
    
        # the bird stays in the same x position, so bird.x is a constant
        # center bird on screen
        bird = Bird(50., float(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
                    (images['bird-wingup'], images['bird-wingdown']))
    
        pipes = deque()
    
        frame_clock = 0  # this counter is only incremented if the game isn't paused
        score = 0
        done = False
        paused = False
        
        def writeData(xDist, yDist, yVel, didFlap):
            xDist = round(xDist,7)
            f = open("data.txt", "a") # input_data or test_labels
            f.write("%s,%s,%s,%s\n" % (xDist, yDist, yVel, didFlap))
            f.close()
            
            print("xDist = %s\nyDist = %s\nyVel = %s\ndid flap = %s\n" % (xDist, yDist, yVel, didFlap))
    
        def checkFlap():
            did = "0"
            if kb.is_pressed('space'):
                if bird.yVel > 0:
                    bird.yVel = - 8
                    did = "1"
    
            return did
    
        def getFlap(x,y,yVel):
            global loopCount
            flap = 0.0
            if loopCount % 2 == 0:
                inpu = np.array([x,y,yVel])
                inpu = inpu.reshape(1,3)
        
                flap = model.predict(inpu)
                flap = round(flap[0,0],3)
                
                if flap >= .5:
                    print("%s\tperdicted: %s\n" % (inpu[0:1],flap))
                    return True
                else:
                    return False
                    
                loopCount += 1
                
        while not done:
            clock.tick(FPS)
            # Handle this 'manually'.  If we used pygame.time.set_timer(),
            # pipe addition would be messed up when paused.
            if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
                pp = PipePair(images['pipe-end'], images['pipe-body'])
                pipes.append(pp)
    
            p = pipes[0]
            
            flapped = ''
            if AI:
                if getFlap(p.getXdist(), p.getTopBottomY() - bird.getY(), bird.getyVel()):
                   bird.yVel = -8
                   flapped='1'
                else:
                    flapped='0'
                
                if collectData == 'ai':
                    writeData(round(float(p.getXdist()),7), round(p.getTopBottomY() - bird.getY(),7), bird.getyVel(), flapped)
                   
            if collectData == 'user':
                writeData(round(float(p.getXdist()),7), round(p.getTopBottomY() - bird.getY(),7), bird.getyVel(), checkFlap())
                
            checkFlap() # checks if user pressed space, then maked bird flap
            
            if paused:
                continue  # don't draw anything
    
            # check for collisions
            pipe_collision = any(p.collides_with(bird) for p in pipes)
            
            if pipe_collision or 0 >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                done = True
    
            for x in (0, WIN_WIDTH / 2):
                display_surface.blit(images['background'], (x, 0))
    
            while pipes and not pipes[0].visible:
                pipes.popleft()
            
            for p in pipes:
    
                p.update()
                display_surface.blit(p.image, p.rect)
                
    
            bird.update()
            display_surface.blit(bird.image, bird.rect)
    
            # update and display score
            for p in pipes:
                if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                    score += 1
                    p.score_counted = True
    
            score_surface = score_font.render(str(score), True, (255, 255, 255))
            score_x = WIN_WIDTH/2 - score_surface.get_width()/2
            display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))
    
            pygame.display.flip()
            frame_clock += 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    break
            
        print('Game over! Score: %i' % score)
        #df.to_csv('data.csv')
        pygame.quit()
    

   

    if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
        main()












