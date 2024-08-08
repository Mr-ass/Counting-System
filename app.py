import argparse
from flask import Flask, request, render_template, redirect, url_for
import torch
import os
import cv2
from detect import main, out_num

app = Flask(__name__)

#读取图片
def return_img_stream(img_local_path):
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


@app.route('/', methods=['GET', 'POST'])
def upload():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        f = request.files['file']
        # filename带后缀，name不带后缀
        filename = f.filename
        name, extension = os.path.splitext(filename)
        file_path = os.path.join(os.getcwd(), "upload_img/", filename)
        #保存上传图片
        f.save(file_path)

        #检测物体
        opt_counting = parse_opt_counting(file_path)
        main(opt_counting)
        #获取检测完的图片并数据分析
        img_path = os.path.join(os.getcwd(), 'runs/detect/exp/', filename)
        detect_stream = return_img_stream(img_path)
        num = out_num()
        density = round(num / 100, 2)
        if density > 0.93:
            grade = "High"
        elif density < 0.66:
            grade = "Low"
        else:
            grade = "Middle"
        return render_template('index.html', detect_stream=detect_stream, num=num, density=density, grade=grade)
    return redirect(url_for('upload'))


def parse_opt_counting(opt_counting):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/snc-best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=opt_counting, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='models/SNc-yolov5s.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--line-thickness', default=10, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
