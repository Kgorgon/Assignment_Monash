import cv2 as cv
import argparse
import time
import logging

log = logging.getLogger(__name__)
# definition of point location
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
# combination of pose pairs
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

def predict(proto,model,src_img, dst_img,img_width=368, img_height=368, threshold=0.3):
  net = cv.dnn.readNetFromCaffe(proto, model)
     
  img = cv.imread(src_img)
  width = img.shape[1]
  height = img.shape[0]

  inp = cv.dnn.blobFromImage(img, 1.0 / 255, (img_width, img_height), (0, 0, 0), swapRB=False, crop=False)
  net.setInput(inp)
  out = net.forward()
  
  points = []
  results = []
  for i in range(len(BODY_PARTS)):
      # Slice heatmap of corresponding body's part.
      heatmap = out[0, i, :, :]

      # Originally, we try to find all the local maximums. To simplify a sample
      # we just find a global one. However only a single pose at the same time
      # could be detected this way.
      _, conf, _, point = cv.minMaxLoc(heatmap)
      x = (width * point[0]) / out.shape[3]
      y = (height * point[1]) / out.shape[2]

      # Add a point if it's confidence is higher than threshold.
      points.append((int(x), int(y)) if conf > threshold else None)
      results.append((int(x), int(y),float(conf)) if conf > threshold else None)

  for pair in POSE_PAIRS:
      partfrom = pair[0]
      partto = pair[1]
      assert(partfrom in BODY_PARTS)
      assert(partto in BODY_PARTS)

      id_from = BODY_PARTS[partfrom]
      id_to = BODY_PARTS[partto]
      if points[id_from] and points[id_to]:
          cv.line(img, points[id_from], points[id_to], (255, 74, 0), 3)
          cv.ellipse(img, points[id_from], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
          cv.ellipse(img, points[id_to], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
          cv.putText(img, str(id_from), points[id_from], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)
          cv.putText(img, str(id_to), points[id_to], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)
        
  cv.imwrite(dst_img, img)
  return results

def main():
    try:
        log.info("Loading pose detection model...")
        proto = "./pose_deploy_linevec.prototxt"
        model = "./pose_iter_440000.caffemodel"
        # Predict on an image
        image_path = './test.jpg'  # Replace with your image path
        output_image = 'test_with_keypoints.jpg'
        result = predict(proto,model,image_path,output_image)
        log.info(result)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
