I’ve kept this main page brief so you can see the results quickly.

main.py: The code for the project the main logic.
keypoint_metrics.csv: The raw data extracted from the video.
overlay_video.mp4: Visual proof of the tracking
Documentation/: 
This folder contains a Detailed Technical Report covering my full observation of the jitter/occlusion issues,
my logic for splitting data (70/15/15), and my long-term plan to adapt this for professional cricket academies.

Note: Real-world data is messy. You’ll notice some jitter due to the practice netting, but I’ve built the pipeline to prioritize
"real-world phone footage" over "perfect lab data" because that’s what coaches actually use.


**My approach**
I did not want to treat this as just another assignment to submit and move on from. While working
on it, I kept thinking that if I get selected, this kind of work would be part of real projects, not just
practice. So I tried to build something that could work in realistic conditions instead of only
looking good on paper.

I spent some time looking at different pose estimation models like YOLO,RTMpose MediaPipe. I
finally chose MediaPipe because it works reasonably well even when the data is limited, which is
the case in an assignment, and because it handles phone recorded videos more reliably.

While working with the video, it became clear that real world data is not clean, and that is normal.
The practice net in the video creates a lot of visual noise, and sometimes the model gets confused
and predicts joints where they are not actually visible. Instead of ignoring this issue, I tried to
handle it in a simple way by smoothing the pose data over multiple frames so that sudden false
detections do not affect the final results too much.

Overall, the aim was not to build something perfect, but to understand the problems that come with
real cricket videos and try to deal with them in a practical way. I wanted the final output to be
something that makes sense and could be useful in a real setting, even if it is not flawless.

**MediaPipe Pose**
MediaPipe Pose was selected after evaluating alternatives such as YOLO Pose and RTMPose.
Reasons for choosing MediaPipe:
  Works reliably on phone-recorded videos
  Performs well even with limited data
  Handles partial occlusion from practice nets better than many models
  Provides 33 detailed body landmarks, including foot keypoints
  Stable and usable out of the box without training or fine-tuning
Since the goal was movement analysis rather than model training, MediaPipe was the most suitable choice for this assignment.

 **Metrics Defined**
The following metrics were extracted from the pose data:
   Lead Elbow Angle
Helps evaluate bat control and straight bat path during the shot.
   Lead Knee Stability
Indicates balance and power generation through the front leg.
   Trunk Lean
Measures body balance and weight transfer during the stroke.

**Observations & Limitations**
While analyzing the video, several real-world challenges were observed:
Jitter: Small frame-to-frame keypoint fluctuations
Occlusion: Far-side joints hidden by the body in side view
Incorrect keypoints: Practice net sometimes mistaken for joints
Motion blur: Fast bat swings reduce wrist and bat clarity
Single-view bias: Lack of depth information from one camera angle

**Improvement Plan**
With more time and data, the following improvements would be made:
Collect more cricket-specific videos across players and environments
Use multiple camera angles for better depth understanding
Combine outputs from multiple pose models
Apply adaptive temporal filtering for fast movements
Evaluate improvements using stability and consistency metrics
