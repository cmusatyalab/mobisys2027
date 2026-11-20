Modules:
    "github.com/cmusatyalab/steeleagle/sdk" v0.1.3

Imports:
    "github.com/cmusatyalab/steeleagle/sdk/dsl/actions"
    "github.com/cmusatyalab/steeleagle/sdk/dsl/events"
    "github.com/cmusatyalab/steeleagle/sdk/dsl/datatypes"
    "github.com/cmusatyalab/steeleagle/sdk/params"
    "github.com/cmusatyalab/steeleagle/sdk/enums"

Actions:
    actions.SetGimbalPose set_gimbal_pose_1(Pose=datatypes.Pose(Pitch=-60.0, Roll=0.0, Yaw=0.0), AngleMode=enums.AngleModeAbsolute, Frame=enums.ReferenceFrameBody)
    actions.Patrol patrol_1(Area=params.Polygon_Polygon_1, Altitude=10.0, AltitudeMode=enums.AltitudeModeRelative, Pattern=actions.PatrolModeSurvey, SurveySpacing=10.0, SurveyHeading=20.0, Speed=3.0, AngularSpeed=120.0)

Mission:
    Start set_gimbal_pose_1
    During set_gimbal_pose_1:
        done -> patrol_1
