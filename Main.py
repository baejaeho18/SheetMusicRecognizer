#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import os
import numpy as np
import functions as fs
import modules

# 이미지 불러오기
print("Processing image_0.. \n")
image_0 = cv2.imread("C:/Users/USER/git/SheetMusicRecognizer/image/music2.jpg")
print("Processing image_1.. \n")
image_1 = modules.remove_noise(image_0)
# 2. 오선 제거
print("Processing image_2.. \n")
image_2, staves = modules.remove_staves(image_1)

# 3. 악보 이미지 정규화
print("Processing image_3.. \n")
image_3, staves = modules.normalization(image_2, staves, 10)

# 4. 객체 검출 과정
print("Processing image_4.. \n")
image_4, objects = modules.object_detection(image_3, staves)

# 5. 객체 분석 과정
print("Processing image_5.. \n")
image_5, objects = modules.object_analysis(image_4, objects)

# 6. 인식 과정
print("Processing image_6.. \n")
image_6, key, beats, pitches = modules.recognition(image_5, staves, objects)

# 이미지 띄우기
cv2.imshow('image', image_6)

print(key)
print(beats)
print(pitches)

with open("music.txt", 'a') as f:
    f.write(beats)
    f.write(pitches)

k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




