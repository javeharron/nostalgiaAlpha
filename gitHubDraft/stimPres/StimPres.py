
import os
import pyautogui
import time
import random
import keyboard
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets, LogLevels
from brainflow.data_filter import DataFilter


BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.ip_port = 6789
params.ip_address = "192.168.4.1"
board = BoardShim(BoardIds.CYTON_DAISY_WIFI_BOARD, params)

board.prepare_session()

# Check sample rate, change to 250 Hz
board.config_board("~~")
board.config_board("~6")

# Check board mode, change to Marker mode
board.config_board("//")
board.config_board("/4")


for i in range(1,9):
    board.config_board("x" + str(i) + "000000X")

daisy_channels = "QWERTYUI"

for i in range(0,8):
    board.config_board("x" + daisy_channels[i] + "000000X")

num_tests = 24; 
wait_slide = '15' # slide number of wait slide
begin_test_slide = '2' # slide number of begin test slide
blank_slide = '16' # slide number of wait slide


# Corresponds to slides 3-14

slides =[
    ['boat_1', 2],
    ['dog_1', 2],
    ['apple_1', 2],
    ['cat_1', 2],
    ['bowling_1', 2],
    ['banana_1', 2],
    ['boat_2', 2],
    ['dog_2', 2],
    ['apple_2', 2],
    ['cat_2', 2],
    ['bowling_2', 2],
    ['banana_2', 2],
    ]

# Tests if each block was repeated 2 times for 10 total tests per image
test_slides =[
    ['boat_1', 0],
    ['dog_1', 0],
    ['apple_1', 0],
    ['cat_1', 0],
    ['bowling_1', 0],
    ['banana_1', 0],
    ['boat_2', 0],
    ['dog_2', 0],
    ['apple_2', 0],
    ['cat_2', 0],
    ['bowling_2', 0],
    ['banana_2', 0],
    ]

absolute_path = os.path.dirname(__file__)
relative_path = "stimulus.pptx"
full_path = os.path.join(absolute_path, relative_path)

#Mac OS Start
#os.start("name" + full_path) # Potential Correct MAC Start

# PC OS Start
os.startfile(full_path)

keyboard.wait('x')

# Sets the powerpoint to fullscreen
pyautogui.hotkey(full_path,'f5')

time.sleep(2)
# Move to the Rules Slide
pyautogui.press('enter')
# Sleep to allow reading of the rules
time.sleep(10)

# one_block presents visual stimulus, then conducts ten EEG tests separated into 2 second intervals
# slides_place is the index in the slides of the phoneme being tested
def one_block(slides_place):
    #image slides  begin at slide 3
    current_slide = slides_place + 3
    str_current_slide = str(current_slide)

    # keypress for the slide number, if slide number has two digits press the digit in the ones place
    pyautogui.press(str_current_slide[0])
    if(current_slide >= 10):
        pyautogui.press(str_current_slide[1])
    pyautogui.press('enter') 
    
    time.sleep(5) # time for subject to analyse picture
    
    board.start_stream()
    # repeat test procedure 10 times
    for i in range(1,11):

        # test_slides incremented by 1
        test_slides[slides_place][1] += 1

        #load wait slide
        pyautogui.press(wait_slide[0])
        pyautogui.press(wait_slide[1])
        pyautogui.press('enter')

        # 1 second buffer
        time.sleep(1)

        # load blank slide
        pyautogui.press(blank_slide[0])
        pyautogui.press(blank_slide[1])
        pyautogui.press('enter')
        board.insert_marker(i)
        
        time.sleep(2)
        board.insert_marker(i)             
    #load wait slide
    pyautogui.press(wait_slide[0])
    pyautogui.press(wait_slide[1])
    pyautogui.press('enter')

    data = board.get_board_data()
    board.stop_stream()
    
    ## IMPORTANT Change Initials for Each Test Subject
    initials = 'SM001'

    naming_convention = initials + '_' + str(slides_place) +'_3'

    # Print File
    DataFilter.write_file(data, naming_convention + '.txt', 'w')


    # 2 second buffer
    time.sleep(2)

    return

# generates random image and tests it if it has not reached the trial limit (10 trials for each image)
while num_tests > 12:

    # pseudo-randomly generated number in range for first image variation
    slides_place = random.randint(0, 5)

    # test the image if slides counter has not reached zero
    if(slides[slides_place][1] > 0):
        slides[slides_place][1] -= 2
        num_tests -= 2
        one_block(slides_place)

while num_tests > 0:
    # pseudo-randomly generated number in range for second image variation
    slides_place = random.randint(6, 11)

    # test the image if slides counter has not reached zero
    if(slides[slides_place][1] > 0):
        slides[slides_place][1] -= 2
        num_tests -= 2
        one_block(slides_place)

# board.release_session()
    
#prints number of blocks conducted for every phoneme
for i in range(0,11):

    print(test_slides[i][1])

pyautogui.hotkey('esc')
