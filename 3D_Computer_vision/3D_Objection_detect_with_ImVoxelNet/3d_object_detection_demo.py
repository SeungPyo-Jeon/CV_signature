import mmcv
import mmengine
import torch
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData
from mmdet3d.visualization import Det3DLocalVisualizer
from mmengine.config import Config
import numpy as np
from mmdet3d.structures import Det3DDataSample, DepthInstance3DBoxes, Box3DMode
if __name__ == "__main__":
    config_file = './test/configs/imvoxelnet_2xb4_sunrgbd-3d-10class.py'
    checkpoint_file = './test/ckpt/imvoxelnet_4x2_sunrgbd-3d-10class_20220809_184416-29ca7d2e.pth'
    device = 'cuda:0'

    img_path = "./test/data/sunrgbd/000017.jpg"
    info_path = "./test/data/sunrgbd/sunrgbd_000017_infos.pkl"

    # 1. 모델과 설정을 로드합니다.
    cfg = Config.fromfile(config_file)
    model = init_model(cfg, checkpoint_file, device=device)
    #model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromFile' # 파이프라인 타입을 수정합니다.

    # 2. 이미지와 pkl 파일에서 메타정보를 직접 로드합니다.
    img = mmcv.imread(img_path)
    data_info = mmengine.load(info_path)
    
    # 3. 모델의 입력 형식에 맞게 데이터를 구성합니다.
    #    Det3DDataSample 객체를 사용하여 이미지와 메타정보를 래핑합니다.
    data_sample = Det3DDataSample()
    metainfo = {}
    # pkl 파일에서 프로젝션 행렬(depth2img)을 가져와 cam2img 키로 저장합니다.
    # 모델은 이 'cam2img' 키를 사용하여 프로젝션 행렬을 찾습니다.
    print( data_info)
    projection_matrix = data_info['data_list'][0]['images']['CAM0']['depth2img']
    metainfo['depth2img'] = np.array(projection_matrix)
    metainfo['img_shape'] = img.shape[:2]
    metainfo['box_type_3d'] = DepthInstance3DBoxes
    metainfo['box_mode_3d'] = Box3DMode.DEPTH
    data_sample.set_metainfo(metainfo)
    print(cfg)
    img_mean = cfg.model.data_preprocessor.mean
    img_std = cfg.model.data_preprocessor.std
    # 모델의 데이터 전처리기(data_preprocessor)를 사용하여 입력을 포맷팅합니다.
    #data = dict(inputs=[img], data_samples=[data_sample])
    #data = model.data_preprocessor(data, True)
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    
    # 4.3. 이미지를 정규화합니다.
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    
    mean_tensor = torch.tensor(img_mean).view(3, 1, 1)
    std_tensor = torch.tensor(img_std).view(3, 1, 1)
    img_tensor = (img_tensor - mean_tensor) / std_tensor # 정규화 수행

    # 4.4. 배치(batch) 차원을 추가하고 GPU로 보냅니다.
    img_tensor = img_tensor.unsqueeze(0).to(device) # (1, C, H, W)

    # 5. 모델이 기대하는 최종 입력 딕셔너리를 구성합니다.
    data = {
        'inputs': {'imgs': img_tensor},
        'data_samples': [data_sample]
    }
    model.bbox_head.test_cfg.score_thr = 0.1
    print(data)
    # 4. 추론을 실행합니다.
    with torch.no_grad():
        results = model.forward(mode='predict', **data)

    # 5. 결과를 시각화합니다.
    result_sample = results[0]
    pred_instances = result_sample.pred_instances_3d
    
    if 'bboxes_3d' in pred_instances and len(pred_instances) > 0:
        print(f"✅ Found {len(pred_instances)} objects.")
        bboxes_3d = pred_instances.bboxes_3d.detach().cpu()
        
        visualizer = Det3DLocalVisualizer()
        visualizer.set_image(img)
        
        # 결과에 포함된 프로젝션 행렬을 사용합니다.
        depth2img_matrix = result_sample.metainfo['depth2img']

        visualizer.draw_proj_bboxes_3d(bboxes_3d, result_sample.metainfo)
        visualizer.show()
    else:
        print("❌ No 3D objects were detected in the image.")
