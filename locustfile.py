import base64
import json
import os
import uuid
from locust import HttpUser, task, between

class CloudPoseUser(HttpUser):
    wait_time = between(1, 3)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_files = []
        # 加载测试图像
        self.load_test_image("test.jpg")
    
    def load_test_image(self, image_path):
        """加载单个测试图像"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                self.image_files.append(image_data)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    @task(2)
    def test_pose_json_api(self):
        """测试JSON API端点"""
        if not self.image_files:
            return
            
        image_data = self.image_files[0]
        encoded_string = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "id": str(uuid.uuid4()),
            "image": encoded_string
        }
        
        self.client.post("/api/pose_estimation", json=payload)
    
    @task(1)
    def test_pose_image_api(self):
        """测试图像API端点"""
        if not self.image_files:
            return
            
        image_data = self.image_files[0]
        encoded_string = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "id": str(uuid.uuid4()),
            "image": encoded_string
        }
        
        self.client.post("/api/pose_estimation_annotation", json=payload)