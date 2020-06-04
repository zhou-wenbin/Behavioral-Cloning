# **Behavioral Cloning** 

## Writeup

### This writeup serves as the explanation of how I successfully run the car in autonomous mode within one loop without accident.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/histogram_origin.png
[image2]: ./output_images/left.png
[image3]: ./output_images/center.png
[image4]: ./output_images/right.png
[image5]: ./output_images/center_crop.png
[image6]: ./output_images/crop.png
[image7]: ./output_images/corner_resize.png
[image8]: ./output_images/origin.png
[image9]: ./output_images/corner_yuv.png
[image10]: ./output_images/flip.png
[image11]: ./output_images/his_blance.png
[image12]: ./output_images/his.png



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4




### Data collection and balancing

#### 1. I have tried various data set: 
1. Self-collected data from simulator 
2. Default data from Udacity
3. Recovery data from road edge to road center turn.
4. Zigzag running data

However, among all these data, only Default data from Udacity worked well. I think it is the reason that the data was collected in joystick. By keyborads it is very diffculty to collect high quality data with continuous steering angle changing data. This experience has taught me that a good quality data makes difference in the deep learning model performance. 

#### 2. Label distribution

The first thing to analyze is to check the label distribution. In the deep learning model the steering angles are the label that need to be learnt from the data. After drawing the histogram of steering angles distributions below, we can see that there zero angle consist most of the data. 


* The histogram of the original data

![alt text][image12]  
* The samples from the original data

![alt text][image8]  


#### 3. Data balancing

We balance the data by randomly picking up 25% of the those data that the angles are zero. In the experiment, we see that with too many portion of zero angle steering data, the car are more likely to run stright. And the balanced data plays an important role in predicting the steering angle in the later on model training. The following function does the balancing:

```python
def collect_data(path):
    lines=[]
    drive_log_path = path + "driving_log.csv"
    data_path = path + "IMG/"
    with open(drive_log_path) as csvfile:
        reader=csv.reader(csvfile)
        next(csvfile)
        for line in reader:
            lines.append(line)
    for line in lines:
        if float(line[3]) != 0:
            image_process(line,data_path)       
        else:
            prob = np.random.uniform()
            if prob <= 0.25: #adjust the ratio to balance the data
                image_process(line,data_path)
```


### Image processing

1. In this project, I have done several image processing, such as 
2. Image crop: to get rid of redundant information like trees, sky and front of car body
3. Image resize: to fit into the deep learning model designed by Nvidia
4. Image flip: To balance the turning right data
5. Image color channel change: to standout the lane in the image.


* Image crop
![alt text][image6]  
* Image resize to 64x64x3      
![alt text][image7]  
* Color channel change from RGB to YUV
![alt text][image9]  
* Image flip, in the same time reverse the sign of associated angle.
![alt text][image10]  

After we the image process, we are able to balance the data with a balanced steering angles set

* Blances data
![alt text][image11]  



### Deep model construction

I tested NVIDIA  architecture but I did not do the normalization part for the data but it went well. I used keras API to build the model, compared to tensorflow we used in the previous section, keras API is much more easy to handle and easy to check the input-output parameters


```python
model = Sequential()
model.add(Lambda(lambda x: x, input_shape=(64, 64, 3)))
model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv1'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(1, 1), activation='relu', name='Conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(128, 2, 2, border_mode='same', subsample=(1, 1), activation='relu', name='Conv4'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', name='FC1'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', name='FC2'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', name='FC3'))
model.add(Dense(1))
```
In the above layers, dropout layer plays an important role in proventing the model to be overfitting. 
Besides the layers, I use Adam optimizer
```python
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer='adam')
```
In order to save the memory and train the model faster, I use ``fit_generator`` API to iterate the data within a batch size. With its help my training time reduces a lot. 

#### Parameters that to be fine-tuned are shown in the following table

After tried many times of fine-tuning the following two parameter, I found that with 9 Epochs and 64 batch size the model could be successful.


| parameters       |    value  | 
|:-------------:|:-------------:| 
| Epochs     | 9     | 
| Batch_size    | 64     | 

At the end I will share the project video that needs to be submitted here.

[![Click to check the project video](https://www.youtube.com/watch?v=XOdgMOJydZ8)](https://www.youtube.com/watch?v=XOdgMOJydZ8) 

### Reflections
As we can see from the video that the car was swinging in the road even though the car did not run out of the track. This is due to the right and left images data with correction on the steering angle that were used in the training. And I have tuned the correction value for many rounds and 0.25 was the value that makes the car run inside the lane but still now stable enough.  I have also spent more than two weeks to train the model and process the image under various method, like change the brightness change the size, but those only helps were presented above in this note. 

Among all the 4 projects that I have done, this one is most challenging for me because I have failed many times in the big turn corner place where the car were always not able to turn big enough to stay inside the lane. But after many trials and errors, I am able to ensure the car runing insde the lane. Even though I am not very satisfied with the performance now but I have no other solutions for the moment now I will first submit first to wait for better suggested solutions. 




```python

```
