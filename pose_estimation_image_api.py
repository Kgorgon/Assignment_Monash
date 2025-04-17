import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

app = FastAPI(title="CloudPose API", description="Pose Estimation Image API Service")

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
net = cv.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)


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
        image = cv.imdecode(np.frombuffer(image_data, np.uint8), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解码失败: {str(e)}")


def encode_image(image: np.ndarray) -> str:
    """
    将OpenCV图像编码为base64字符串
    
    Args:
        image: OpenCV图像
        
    Returns:
        base64编码的图像字符串
    """
    try:
        # 编码图像为JPEG格式
        _, buffer = cv.imencode('.jpg', image)
        # 转换为base64
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像编码失败: {str(e)}")


def process_image_with_annotation(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    处理图像并添加姿态估计标注
    
    Args:
        image: OpenCV图像
        threshold: 置信度阈值
        
    Returns:
        带有姿态估计标注的图像
    """
    # 创建图像副本以避免修改原始图像
    annotated_image = image.copy()
    height, width = image.shape[:2]
    
    # 预处理图像
    inp = cv.dnn.blobFromImage(image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    
    # 设置模型输入
    net.setInput(inp)
    
    # 获取模型输出
    out = net.forward()
    
    # 获取所有检测到的关键点
    points = []
    
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
        else:
            points.append(None)
    
    # 绘制关键点和连接线
    for pair in POSE_PAIRS:
        part_from = pair[0]
        part_to = pair[1]
        
        id_from = BODY_PARTS[part_from]
        id_to = BODY_PARTS[part_to]
        
        if points[id_from] and points[id_to]:
            # 绘制线条
            cv.line(annotated_image, points[id_from], points[id_to], (255, 74, 0), 3)
            # 绘制关键点
            cv.ellipse(annotated_image, points[id_from], (4, 4), 0, 0, 360, (0, 255, 0), cv.FILLED)
            cv.ellipse(annotated_image, points[id_to], (4, 4), 0, 0, 360, (0, 255, 0), cv.FILLED)
    
    # 添加边界框
    valid_points = [p for p in points if p is not None]
    if valid_points:
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 添加一点边距
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(width, max_x + padding)
        max_y = min(height, max_y + padding)
        
        # 绘制边界框
        cv.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    
    return annotated_image


@app.post("/api/pose_estimation_annotation")
async def pose_estimation_annotation(request: ImageRequest) -> JSONResponse:
    """
    接收base64编码的图像并返回带有姿态估计标注的图像
    
    Args:
        request: 包含ID和base64编码图像的请求对象
        
    Returns:
        包含原始ID和base64编码的标注图像的JSON响应
    """
    try:
        # 解码图像
        image = decode_image(request.image)
        
        # 使用线程池处理图像，以处理并发请求
        annotated_image = await app.state.executor.submit(process_image_with_annotation, image)
        
        # 编码标注图像
        encoded_image = encode_image(annotated_image)
        
        # 构建响应
        response = {
            "id": request.id,
            "image": encoded_image
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
    port = 60081
    
    uvicorn.run(app, host="0.0.0.0", port=port)