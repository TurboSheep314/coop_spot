## About The Project
The goal of this project is to work cooperatively with the Boston Dynamics Spot robot to accomplish a task.

For example, if we had a box of cereal, a bowl, milk, and a spoon on a table, we want to work with Spot to make a bowl of cereal, if a human reaches for the cereal, Spot can identify the human's intention and decide what the next best course of action should be, in this instance, he should grab a bowl so we have something to pour the cereal into.

## Roadmap (Not in order)
### Completed
- [x] Hand Tracking
- [x] Calculating velocity of hand
- [x] April Tag detection on ZED Camera
- [x] Determining closest April Tag
### In Progress/High Priority
- [ ] April Tag detection on Spot
- [ ] Arm Commands for Spot to pick up detected objects
- [ ] Environment Setup
### To Do
- [ ] Bayesian Intention Model
- [ ] LFD Training
- [ ] Cue Generation
- [ ] Strategy Model

## How-to
Open the command prompt
Change directory to `~/ros2_dev`
Type `source ~/ros2_dev/install/setup.bash`

First calibrate running `ros2 launch coop_spot calibrate.launch.py`. Get the transform, e.g.
```
[calibrate-1] [INFO] [1751404962.668303732] [coop_spot.calibrate]: transform is ['0.0972', '-0.2843', '-0.1652'], ['3.0417', '0.7441', '-1.5925']
```
and paste it in `/home/aicaring/ros2_ws/install/zed_wrapper/share/zed_wrapper/launch/zed_camera.launch.py`.

To launch the program, connect to the Spot Wi-Fi (`spot-BD-12140002`), and type `ros2 launch coop_spot coop_spot.launch.py`

If you make edits to the program, type `colcon build --packages-select coop_spot` to update it for launch

We use github for version control, if you want to make a commit, change the directory to `~/ros2_dev/src/coop_spot`, type `git commit`, type your commit message, then type `:wq` to save and quit.

To push the commits, type `git push coop_spot`

### Android tablet login
- user: `user`
- password: `sdix4zo439i0`