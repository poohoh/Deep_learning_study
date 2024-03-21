import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

'''
Coco val 2017 데이터셋에 대한 RetinaNet의 Evaluation
이 파일을 실행시키는 위치에 datasets 디렉토리가 존재해야 하고, 그 구조는 다음과 같음
datasets - coco - annotations/instances_val2017.json
         |
         - val2017 - 이미지 파일들
'''

if __name__ == "__main__":
    '''
    # COCO 데이터셋 등록
    register_coco_instances("coco_val", {}, "../../datasets/coco/annotations/instances_val2017.json", "../../datasets/coco/val2017")

    # Config 불러오기
    cfg = get_cfg()
    cfg.merge_from_file("../configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "./model_final_5bd44e.pkl"  # 미리 학습된 모델의 가중치 경로
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 예측 결과를 얻기 위한 예측 임계값 설정
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # 클래스 개수 지정

    # # 모델 빌드
    # model = build_model(cfg)

    # DefaultPredictor 생성
    predictor = DefaultPredictor(cfg)

    # Evaluator 생성
    evaluator = COCOEvaluator("coco_val", cfg, False, output_dir="./output/")

    # 검증 데이터셋에 대한 예측 및 평가
    val_loader = build_detection_test_loader(cfg, "coco_val")

    # 객체 검출 성능만을 평가
    evaluator._tasks = ("bbox",)

    inference_on_dataset(predictor.model, val_loader, evaluator)

    # 결과 출력
    print(evaluator.evaluate())
    '''

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # register_coco_instances("coco_val", {}, "../../datasets/coco/annotations/instances_val2017.json",
    #                         "../../datasets/coco/val2017")
    evaluator = COCOEvaluator("coco_2017_val", output_dir="./output/", tasks=("bbox",))
    val_loader = build_detection_test_loader(cfg, "coco_2017_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # print(evaluator.evaluate())

    # im = cv2.imread('./test_image.png')
    # outputs = predictor(im)
    #
    # print(outputs['instances'].pred_classes)
    # print(outputs['instances'].pred_boxes)
    #
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('.', out.get_image()[:, :, ::-1])
    # cv2.waitKey()
    # cv2.destroyAllWindows()