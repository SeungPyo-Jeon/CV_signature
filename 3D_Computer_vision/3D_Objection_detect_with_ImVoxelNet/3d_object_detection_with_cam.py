import mmcv
import cv2
import mmengine
import torch
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData
from mmdet3d.visualization import Det3DLocalVisualizer
from mmengine.config import Config
import numpy as np
from mmdet3d.structures import Det3DDataSample, DepthInstance3DBoxes, Box3DMode

def main():
    config_file = './test/configs/imvoxelnet_2xb4_sunrgbd-3d-10class.py'
    checkpoint_file = './test/ckpt/imvoxelnet_4x2_sunrgbd-3d-10class_20220809_184416-29ca7d2e.pth'
    device = 'cuda:0'

    img_path = "./test/data/sunrgbd/000017.jpg"
    info_path = "./test/data/sunrgbd/sunrgbd_000017_infos.pkl"

    cfg = Config.fromfile(config_file)
    model = init_model(cfg, checkpoint_file, device=device)
    visualizer = Det3DLocalVisualizer()

    data_info = mmengine.load(info_path)

    data_sample = Det3DDataSample()
    metainfo = {}
    projection_matrix = data_info['data_list'][0]['images']['CAM0']['depth2img']
    metainfo['depth2img'] = np.array(projection_matrix)
    metainfo['box_type_3d'] = DepthInstance3DBoxes
    metainfo['box_mode_3d'] = Box3DMode.DEPTH
    metainfo['classes'] = model.dataset_meta['classes']
    img_mean = cfg.model.data_preprocessor.mean
    img_std = cfg.model.data_preprocessor.std
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 5. Per-Frame Preprocessing & Inference ---
        # Update the image shape in metainfo for each frame
        metainfo['img_shape'] = frame_rgb.shape[:2]
        data_sample.set_metainfo(metainfo)

        # Manually preprocess the image (Normalization and Tensor conversion)
        img_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).float()
        mean_tensor = torch.tensor(img_mean).view(3, 1, 1)
        std_tensor = torch.tensor(img_std).view(3, 1, 1)
        img_tensor = (img_tensor - mean_tensor) / std_tensor
        img_tensor = img_tensor.unsqueeze(0).to(device)

    # 5. 모델이 기대하는 최종 입력 딕셔너리를 구성합니다.
        data = {
            'inputs': {'imgs': img_tensor},
            'data_samples': [data_sample]
        }
        model.bbox_head.test_cfg.score_thr = 0.2

        with torch.no_grad():
            results = model.forward(mode='predict', **data)

        result_sample = results[0]
        pred_instances = result_sample.pred_instances_3d
        
        visualizer.set_image(frame_rgb)
        

        if 'bboxes_3d' in pred_instances and len(pred_instances) > 0:
            visualizer.draw_proj_bboxes_3d(
                bboxes_3d=pred_instances.bboxes_3d.cpu(), 
                input_meta=result_sample.metainfo)
            
        vis_frame = visualizer.get_image()
        
        # Convert back to BGR for displaying with OpenCV
        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('Real-time 3D Object Detection', vis_frame_bgr)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # --- 7. Cleanup ---
    print("Closing webcam feed.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()