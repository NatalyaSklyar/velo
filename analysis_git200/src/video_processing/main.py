from build_sample_image import build_sample_image
from run_pipeline import run_full_video_pipeline


def main():
    VIDEO_PATH = "analysis_git200/data/comprec.mp4"
    MODEL_PATH = "model_training/runs/pose_cyclist/single_split/weights/best.pt"
    SAMPLE_PATH = "analysis_git200/data/sample.jpg"

    build_sample_image(
        video_path=VIDEO_PATH,
        out_path=SAMPLE_PATH,
        start_sec=0,
    )

    run_full_video_pipeline(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        out_csv_path="analysis_git200/data/full/all_tracks_test.csv",

        # time window
        start_sec=0,
        end_sec=33*60+29,

        # ROI
        roi_anchor="right",
        vertical_divisor=1.5,

        # trigger: background from sample image
        sample_image_path=SAMPLE_PATH,

        # trigger
        diff_thresh=25,
        motion_ratio=0.02,
        consec_frames=3,
        cooldown_sec=2.0,
        bg_build_sec=2.0,

        # track
        conf_thres=0.75,
        rewind_sec=1.0,
        stop_lost_patience=20,
        max_track_seconds=12.0,

        # video output: enable for test
        # save_video=True,
        # out_video_path="analysis_git200/data/example/debug_test_18-20.mp4",
        # draw_debug=True,

        # trigger pass visualization/logging
        # save_trigger_video=True,
        # trigger_video_path="analysis_git200/data/example/debug_trigger_18-20.mp4",
        # trigger_log_every=150,
    )


if __name__ == "__main__":
    main()
