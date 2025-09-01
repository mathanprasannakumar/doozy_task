# Standard Library
from pathlib import Path
import json 
import argparse
import os
from pathlib import Path
from typing import Dict
from ultralytics import YOLO

from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np

# Third Party
import torch

from happypose.pose_estimators.megapose.inference.pose_estimator import PoseEstimator

# HappyPose
from happypose.toolbox.datasets.object_dataset import RigidObjectDataset
from happypose.toolbox.inference.example_inference_utils import (
    load_detections,
    load_object_data,
    load_observation_example,
    make_detections_visualization,
    make_example_object_dataset,
    make_poses_visualization,
    save_predictions,
)
from happypose.toolbox.inference.types import DetectionsType, ObservationTensor
from happypose.toolbox.inference.utils import filter_detections, load_detector
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_pose_estimator(model_name: str, object_dataset: RigidObjectDataset):
    logger.info(f"Loading model {model_name}.")
    model_info = NAMED_MODELS[model_name]
    pose_estimator = load_named_model(model_name, object_dataset).to(device)
    # Speed up things by subsampling coarse grid
    pose_estimator._SO3_grid = pose_estimator._SO3_grid[::8]

    return pose_estimator, model_info


def run_inference(
    pose_estimator: PoseEstimator,
    model_info: Dict,
    observation: ObservationTensor,
    detections: DetectionsType,
) -> None:
    observation.to(device)

    logger.info("Running inference.")
    data_TCO_final, extra_data = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections,
        **model_info["inference_parameters"],
    )
    print("Timings:")
    print(extra_data["timing_str"])

    return data_TCO_final.cpu()



def draw_axes(image, K, quat, t, axis_length=0.05):
    """
    Draws 3D coordinate axes on the image given pose (quaternion + translation).

    Args:
        image (np.ndarray): Input RGB image.
        K (np.ndarray): Camera intrinsic matrix (3x3).
        quat (list or np.ndarray): Quaternion [x, y, z, w].
        t (list or np.ndarray): Translation [tx, ty, tz].
        axis_length (float): Length of each axis in world units.
    """
    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat(quat).as_matrix()

    # Define 3D axis points in object coordinate system
    axis_points_3d = np.float32([
        [0, 0, 0],                       # origin
        [axis_length, 0, 0],             # X axis
        [0, axis_length, 0],             # Y axis
        [0, 0, axis_length]              # Z axis
    ]).reshape(-1, 3)

    # Convert rotation matrix to Rodrigues vector
    rvec, _ = cv2.Rodrigues(rot_matrix)
    tvec = np.array(t, dtype=np.float32).reshape(3, 1)

    # Assume no distortion
    dist_coeffs = np.zeros((4,1))

    # Project 3D points to 2D
    imgpts, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, K, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw axes
    origin = tuple(imgpts[0].ravel())
    image = cv2.line(image, origin, tuple(imgpts[1].ravel()), (0,0,255), 3) # X axis = Red
    image = cv2.line(image, origin, tuple(imgpts[2].ravel()), (0,255,0), 3) # Y axis = Green
    image = cv2.line(image, origin, tuple(imgpts[3].ravel()), (255,0,0), 3) # Z axis = Blue

    return image


data_dir = os.getenv("HAPPYPOSE_DATA_DIR")
assert data_dir, "Set HAPPYPOSE_DATA_DIR env variable"
example_dir = Path(data_dir)
assert example_dir.exists(), (
    "Example directory not available, follow download instructions"
)

model = "megapose-1.0-RGB"
# Load data
object_dataset = make_example_object_dataset(example_dir)
_,_, camera_calib = load_observation_example(example_dir, load_depth=False)
# Load models
pose_estimator, model_info = setup_pose_estimator(model, object_dataset)

img_data_dir = "data/hex_bolt_30"

model = YOLO("model/finetuned_yolov11seg_aug31.pt")

for img in os.listdir(img_data_dir):

    rgb = cv2.imread(os.path.join(img_data_dir,img))
    result = model(rgb)

    bbox = result[0].boxes.xyxy[0]
    bbox = bbox.to(torch.int32)
    bbox = bbox.tolist()

    detection_data = [{"label": "hex_bolt_30","bbox_modal": bbox}]

    with open("object_data.json","w") as f:
        json.dump(detection_data,f)
    
    observation = ObservationTensor.from_numpy(rgb, None, camera_calib.K).to(device)

    detections = load_detections(example_dir).to(device)

    output = run_inference(pose_estimator, model_info, observation, detections)
    save_predictions(output, example_dir)


    out_filename = "object_data_inf.json"
    object_datas = load_object_data(example_dir / "outputs" / out_filename)


    prediction_file_path = "outputs/object_data_inf.json"
    camera_data_path = "camera_data.json" 
    prediction_file_path  = Path(prediction_file_path) 
    camera_data_path = Path(camera_data_path)

    predictions  = json.loads(prediction_file_path.read_text())

    quat = predictions[0]['TWO'][0]
    trans = predictions[0]['TWO'][1]
    print("quaternion :",quat)
    print("translation  :",trans)

    camera_data = json.loads(camera_data_path.read_text())
    K = np.array(camera_data['K'])
    print("Calibration matrix: ",K)

    annotated_img = draw_axes(rgb,K,quat,trans)
    cv2.imwrite(f"pose_output/{img}",annotated_img);