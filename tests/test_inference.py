import os
import subprocess

def test_inference_pipeline():
    """
    Runs the inference pipeline on a sample video.
    Requires:
    - tests/sample_video.mp4
    - models/
        - posture_classifier.h5
        - posture_scaler.pkl
        - standing_fall_model.h5
        - standing_scaler.pkl
    """
    sample_video = "tests/sample_video.mp4"
    model_dir = "models"
    yolo_weights = "yolov8n-pose.pt"

    # Check that required files exist
    assert os.path.exists(sample_video), "Sample video not found."
    assert os.path.exists(os.path.join(model_dir, "posture_classifier.h5")), "Missing posture model."
    assert os.path.exists(os.path.join(model_dir, "posture_scaler.pkl")), "Missing posture scaler."

    # Run the inference script using subprocess
    result = subprocess.run([
        "python", "src/inference.py",
        "--video", sample_video,
        "--model_dir", model_dir,
        "--weights", yolo_weights
    ], capture_output=True, text=True)

    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)

    assert "Detected posture" in result.stdout, "Posture detection failed."
    assert "Fall Detection" in result.stdout, "Fall detection output missing."

if __name__ == "__main__":
    test_inference_pipeline()
