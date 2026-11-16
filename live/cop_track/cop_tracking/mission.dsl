Modules:
    "github.com/cmusatyalab/steeleagle/sdk" v0.1.2

Imports:
    "github.com/cmusatyalab/steeleagle/sdk/dsl/actions"
    "github.com/cmusatyalab/steeleagle/sdk/dsl/events"
    "github.com/cmusatyalab/steeleagle/sdk/dsl/datatypes"
    "github.com/cmusatyalab/steeleagle/sdk/params"
    "github.com/cmusatyalab/steeleagle/sdk/enums"

Actions:
    actions.SetGimbalPose set_gimbal_pose_1(Pose=datatypes.Pose(Pitch=-45.0, Roll=0.0, Yaw=0.0), AngleMode=enums.AngleModeAbsolute, Frame=enums.ReferenceFrameBody)
    actions.Track track_1(Camera=params.Camera_rgb, Engine=DetectionEngine_local_cop_tracking, Class="robotic dog", LostDuration=3.0, TargetAreaRatio=0.1, YawSpeed=180.0, PitchSpeed=20.0, MoveSpeed=0.0)

Mission:
    Start set_gimbal_pose_1
    During set_gimbal_pose_1:
        done -> track_1
