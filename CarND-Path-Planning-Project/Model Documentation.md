# Model Documentation

The model start from line ??? to line ??? in `src/main.cpp`. The code has the following parts:

1. **Predictio**: line 253 to line 298
2. **Behavior**: line 301 to line 326
3. **Trajectory Generation**: line

I find it easier to put everything into the main file for better navigation. I also find it important to isolate these three pieces as they perform different things. I am going to walk through each of them separately below.

## Prediction
We collect the data from the outside and to ensure we can do the following:

1. a car is infront of us？ `car_ahead`
2. a car is to our right so that we cannot change to the right line safely? `car_right`
3. a car is to our left so that we cannot change to the left line safely? `car_left`

Here are the steps:

1. setting all the signals above to false.
2. iterate through each cars from `sensor_fusion`.
    1. if the car is not in lane 0, 1, or 2, ignore.
    2. calculate the other car's position `s`, with the future "priced in".
    3. If the car is in front of us and the distance is less than 30 meters, set `car_ahead` to `TRUE`.
    4. If the car is +/- 30 meters from us to our left/right, we set `car_right` or `car_left` to `TRUE`.

## Behavior

Here we want to answer the following question:

1. Should we change lane?
2. Should we speed up or slow down?

Moreover, we pre-define the `MAX_SPEED` to drive being 49.5 m/s (the closer the better), and the `MAX_ACC` to be .224 m^2/s.

Here are the steps:
1.  If there is a car in front of us:
    1. change lane if possible
    2. otherwise, slow down.
2. If there is no car in front of us:
    1. change to the front lane if possible.
    2. speed up if possible.

## Trajectory Generation

The goal of trajectory generation is to come up with a path that is as smooth as possible given the decision made before.

Here are the steps:

1. keep track of the previous 2 points trajectory (or car's position if there is none)
2. set up 3 target points in the future.
3. create a spline using the external `spline.h` pacakage.
4. We keep using the points from `previous_path_x` and add the remaining 50 - prev_size points as fast as possible to the targerts, according to the spline.
