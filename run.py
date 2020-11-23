from datetime import datetime
import os
import cv2
import numpy as np
import threading
import tensorflow as tf
from skimage import metrics as ssim
from apscheduler.schedulers.background import BackgroundScheduler
from yolo.image import detect_image

from flask import Flask, jsonify,  current_app
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
os.listdir(path='.')


from flask import Flask
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
ma = Marshmallow(app)

# This class is use to create table fields in database
class ImageModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(300), nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())


class ImageSchema(ma.ModelSchema):
    class Meta:
        model = ImageModel()


@app.route('/')
def hello_world():
    img = ImageModel.query.all()
    img_schema = ImageSchema(many=True)
    output = img_schema.dump(img)
    return jsonify({'Image': output})




# Get all today data
@app.route('/api/get_today')
def get_today():
    todays_datetime = datetime(datetime.today().year, datetime.today().month, datetime.today().day)
    image = ImageModel.query.filter(ImageModel.date_posted >= todays_datetime)
    image_schema = ImageSchema(many=True)
    output = image_schema.dump(image)
    return jsonify({'Image': output})


#Get last ten data
@app.route('/api/get_last_ten')
def get_last_ten():
    image = ImageModel.query.order_by(ImageModel.date_posted.desc()).limit(10)
    image_schema = ImageSchema(many=True)
    output = image_schema.dump(image)
    return jsonify({'Image': output})


#Get all data
@app.route('/api/get_all')
def get_data():
    image = ImageModel.query.order_by(ImageModel.date_posted.desc()).all()
    image_schema = ImageSchema(many=True)
    output = image_schema.dump(image)
    return jsonify({'Image': output})


# This function is used to find error between two images
def mse(original, duplicate):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((original.astype("float") - duplicate.astype("float")) ** 2)
    err /= float(original.shape[0] * original.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


background = None
def load_cam():
    with app.app_context():
        global background
		#***Univeristy cctv
        #cap = cv2.VideoCapture('rtsp://admin:rex6885!@sel312.iptime.org:20004/MOBILE/media.smp') 
		#***Farm cctv
        cap = cv2.VideoCapture('rtsp://admin:a123456789@218.151.33.75:554/Streaming/channels/301') 
        if cap.isOpened():
            try:
                _, img_stream = cap.read()
                copy_img = img_stream.copy()
                gray = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if background is None:
                    background = gray
                ms = mse(background, gray)
                similarity = ssim.structural_similarity(background, gray)
                round_ssim = "{:.2f}".format(similarity)
                if similarity <= 0.80:
					#***This comment block is used send image and get detected obj from yolo
                    # object_with_label = detect_image(copy_img)
                    # if len(object_with_label) >= 1:
                    # for key, val in object_with_label.items():
                    #     print('{} {:.4f}'.format(key, val))
                    f_name = 'img_ssim' + str(similarity) + '.jpg'
                    f_path_name = os.path.join(current_app.root_path, 'static/saved_images/', f_name)
                    cv2.imwrite(f_path_name, copy_img)
                    image_name = ImageModel(image_name=f_name, date_posted=datetime.now())
                    db.session.add(image_name)
                    db.session.commit()
                    background = gray
                else:
                    print("Frame is not changed. MSE: %.2f, SSIM: %.2f" % (ms, similarity))
                key = cv2.waitKey(30) & 0xFF
            except Exception as e:
                print(e)
                cap.release()
            return jsonify({'success': 'success'})
        else:
            print("No")
        return jsonify({'error': 'error'})


#***Load camera in every 5s interval
scheduler = BackgroundScheduler()
scheduler.add_job(func=load_cam, trigger='interval', seconds=5)
scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)