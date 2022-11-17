# Robotics

## KM&EKF - Kalman Filter & Extended Kalman Filter
  - This folder contians variuous implementations for a Turtlebot. This was simulated using the ROS environment. To try the programms, the "kalman_simulation" is needed, and should be placed as a ROS package. 
  
  -> KFBasicMotion.py; This program uses the kalman filter for a basic turtlebot, where the motion is based on a relative position between the beacon and the robot frame. The robot uses spherical wheels and only moves in a straight line. 
  Result:
  ![image](https://user-images.githubusercontent.com/99536660/201900900-33fc0947-1843-4637-9936-a747b2d222b0.png)
  
  To try this, launch in ros exercise_1_spherical_wheels.launch
  
  ->KFWheelDiameter.py; Given the previous model, know we do not know the diameter of the wheels and we want to find it with a KF.
  Result:
  ![image](https://user-images.githubusercontent.com/99536660/201904145-77c56236-2b68-41ff-acc7-a0bc2833b4a5.png)

  To try this, launch in ros exercise_2_wheel_diameter.launch
  
  ->EKF_For_Turtlebot.py; Given the first model, we change the wheels so instead of beeing spherical, they are circular. This enables us two type of motions, rotation and translation. This is a real model of the turtlebot.
  Result:  
  ![image](https://user-images.githubusercontent.com/99536660/201904651-cafa17ba-c6f7-4073-8f82-1086f5ad4e05.png)

  To try this, launch in ros exercise_4_normal_wheels.launch
  
  ->EKF-Distance&Direction.py; This model only uses the simple translation as in the first model. However know it is not based on beacons, but on distance and direction. This adds more noise.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/201904945-eb1d8816-21ed-4413-9097-69447a14329e.png)

  To try this, launch in ros exercise_5_sensor_model.launch
