import base64
import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

app = FastAPI(title="CloudPose API", description="Pose Estimation API Service")

# 创建线程池，用于处理并发请求
executor = ThreadPoolExecutor(max_workers=10)

# 定义人体关键点
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

# 组合姿势对
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# 模型文件路径
PROTO_FILE = "./pose_deploy_linevec.prototxt"
MODEL_FILE = "./pose_iter_440000.caffemodel"

# 全局加载模型
net = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)


class ImageRequest(BaseModel):
    """
    请求模型，包含ID和base64编码的图像
    """
    id: str
    image: str


def decode_image(base64_image: str) -> np.ndarray:
    """
    将base64编码的图像解码为OpenCV图像
    
    Args:
        base64_image: base64编码的图像字符串
        
    Returns:
        解码后的OpenCV图像
    """
    try:
        # 解码base64字符串
        image_data = base64.b64decode(base64_image)
        # 将字节转换为图像
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解码失败: {str(e)}")


def process_image(image: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
    """
    处理图像并进行姿态估计
    
    Args:
        image: OpenCV图像
        threshold: 置信度阈值
        
    Returns:
        包含姿态估计结果的字典
    """
    # 计时：预处理开始
    preprocess_start = time.time()
    
    height, width = image.shape[:2]
    
    # 预处理图像
    inp = cv2.dnn.blobFromImage(image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    
    # 计时：预处理结束
    preprocess_end = time.time()
    speed_preprocess = preprocess_end - preprocess_start
    
    # 计时：推理开始
    inference_start = time.time()
    
    # 模型推理
    net.setInput(inp)
    out = net.forward()
    
    # 计时：推理结束
    inference_end = time.time()
    speed_inference = inference_end - inference_start
    
    # 计时：后处理开始
    postprocess_start = time.time()
    
    # 初始化结果列表
    keypoints_list = []
    boxes = []
    persons_count = 0
    
    # 获取所有检测到的关键点
    points = []
    person_keypoints = []
    
    # 检测人体关键点
    for i in range(len(BODY_PARTS) - 1):  # 不包括背景
        # 切片对应身体部位的热图
        heatmap = out[0, i, :, :]
        
        # 查找全局最大值
        _, conf, _, point = cv.minMaxLoc(heatmap)
        
        # 坐标转换
        x = (width * point[0]) / out.shape[3]
        y = (height * point[1]) / out.shape[2]
        
        # 添加点（如果置信度高于阈值）
        if conf > threshold:
            points.append((int(x), int(y)))
            person_keypoints.append([float(x), float(y), float(conf)])
        else:
            points.append(None)
            person_keypoints.append([0, 0, 0])  # 如果低于阈值，使用0填充
    
    # 如果至少检测到一个关键点，则认为检测到一个人
    if any(point is not None for point in points):
        persons_count = 1
        keypoints_list.append(person_keypoints)
        
        # 创建边界框（使用检测到的关键点的最大和最小坐标）
        valid_points = [p for p in points if p is not None]
        if valid_points:
            x_coords = [p[0] for p in valid_points]
            y_coords = [p[1] for p in valid_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            box_width = max_x - min_x
            box_height = max_y - min_y
            
            # 添加一点边距
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            box_width = min(width - min_x, box_width + 2 * padding)
            box_height = min(height - min_y, box_height + 2 * padding)
            
            # 计算置信度（使用有效关键点的平均置信度）
            confidences = [kp[2] for kp in person_keypoints if kp[2] > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            boxes.append({
                "x": float(min_x),
                "y": float(min_y),
                "width": float(box_width),
                "height": float(box_height),
                "probability": float(avg_confidence)
            })
    
    # 计时：后处理结束
    postprocess_end = time.time()
    speed_postprocess = postprocess_end - postprocess_start
    
    return {
        "count": persons_count,
        "boxes": boxes,
        "keypoints": keypoints_list,
        "speed_preprocess": round(speed_preprocess, 3),
        "speed_inference": round(speed_inference, 3),
        "speed_postprocess": round(speed_postprocess, 3)
    }


@app.post("/api/pose_estimation")
async def pose_estimation(request: ImageRequest) -> JSONResponse:
    """
    接收base64编码的图像并返回姿态估计结果
    
    Args:
        request: 包含ID和base64编码图像的请求对象
        
    Returns:
        包含姿态估计结果的JSON响应
    """
    try:
        # 解码图像
        image = decode_image(request.image)
        
        # 使用线程池处理图像，以处理并发请求
        results = await app.state.executor.submit(process_image, image)
        
        # 构建响应
        response = {
            "id": request.id,
            **results
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """
    应用启动时初始化
    """
    app.state.executor = executor


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时清理资源
    """
    app.state.executor.shutdown()


# 主入口点
if __name__ == "__main__":
    import uvicorn
    
    # 使用端口范围60000-61000中的一个端口
    port = 60080
    
    uvicorn.run(app, host="0.0.0.0", port=port)