import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
from datetime import timedelta
from tensorflow.python import gfile
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
import base64
tf.disable_v2_behavior()
############################################################################
dir = r'./static/images/'
model_dir = r'./ChineseHerbalMedicine/model/tensorflow_inception_graph.pb'


# 对应类别
def labels_datas():
    labels_data = ['鬼针草', '红蓼', '苦艾', '荨麻', '龙牙草',
                   '栀子花', '板蓝根', '狗尾巴草', '车前草', '通泉草',
                   '漆姑草', '鸭舌草', '鸭跖草',
                   '小窃草', '水芹菜', '柴胡', '蒲公英',
                   '半夏', '酢浆草', '溪黄草', '飞蓬',
                   '金银花', '小蓟', '夏枯草', '牵牛花',
                   '马兰', '天胡荽', '紫云英', '凤眼兰',
                   '何首乌', '藜', '麦冬', '黄花菜',
                   '曼陀罗', '苦菜', '垂盆草', '波斯婆婆纳', '野薄荷', '卷耳',
                   '莲子', '棕榈', '苍耳子', '兰花参', '白芷',
                   ' 高丽参', '泽漆', '厚朴', '紫苏', '一串红', '荠菜',
                   '玉竹', '灯笼草', '藿香', '金钱草']
    return labels_data


# 创建空图
def load_pretrainder_inception_v3(model_dir):
    with gfile.FastGFile(model_dir, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


labels_data = labels_datas()
x = tf.placeholder(tf.float32, [None, 2048])
y = tf.placeholder(tf.int64, [None])
fc1 = tf.layers.dense(x, 1024, activation=tf.nn.relu)
h = tf.layers.dense(fc1, 54, activation=tf.nn.softmax)
load_pretrainder_inception_v3(model_dir)
sess = tf.Session()
second_to_last_tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./ChineseHerbalMedicine/bottleneck1/'))


# 标签
def labels_class(data_dir):
    # 读取图片内容
    img_data = gfile.FastGFile(data_dir, 'rb').read()
    batch_feature = sess.run(second_to_last_tensor, feed_dict={'DecodeJpeg/contents:0': img_data})
    batch_features = np.vstack(batch_feature)
    # 调用训练好的模型
    label = sess.run(tf.argmax(h, 1), feed_dict={x: batch_features})  # 预测种类
    data = sess.run(h, feed_dict={x: batch_features})
    probability = np.max(data)
    print(probability)
    return label, probability


############################################################################

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# 输出
@app.route('/')
def hello_world():
    return render_template('index.html')


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/api/upload', methods=['POST'])
def upload():
    """
    api:http://127.0.0.1:5000/api/upload
    发送：type="file" name="file"
    接受：accuracy ：浮点型，准确率
          class_name ： 字符串，类别
          is_object ： 整形，1  代表是中草药类别，0 其他物品
    例子：
        {
          "accuracy": "0.990355",
          "class_name": "曼陀罗",
          "is_object": 1
        }
    """
    if request.method == 'POST':
        # 通过file标签获取文件
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "图片类型：png、PNG、jpg、JPG、bmp"})
        # 当前文件所在路径
        basepath = os.path.dirname(__file__)
        # 一定要先创建该文件夹，不然会提示没有该路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        # 保存文件
        f.save(upload_path)
        folder_name = f.filename  # 上传文件名字
        ################################################################################
        tic = time.time()
        # 图片全路径
        data_dir = os.path.join(dir, folder_name)
        label, probability = labels_class(data_dir)
        class_name = labels_data[label[0]]
        toc = time.time()
        print('all,used{:.5}s'.format(toc - tic))
        ################################################################################
        if probability >= 0.7:
            user = {'accuracy': str(probability), 'is_object': 1, 'class_name': class_name}
        else:
            user = {'accuracy': str(probability), 'is_object': 0, 'class_name': class_name}
        return jsonify(user)
    # 重新返回上传界面
    return render_template('index.html')


@app.route('/cap', methods=['GET'])
def hello_cap():
    return render_template('capture.html')


@app.route('/api/capture', methods=['POST'])
def capture():
    """
    api:http://127.0.0.1:5000/api/capture
    发送：type="base64" name="base64_11"
    接受： accuracy ：浮点型，准确率
          class_name ： 字符串，类别
          is_object ： 整形，1  代表是中草药类别，0 其他物品
    例子：
        {
          "accuracy": "0.990355",
          "class_name": "曼陀罗",
          "is_object": 1
        }
    """
    base64_1 = request.json["base64_11"]
    imgdata = base64.b64decode(base64_1)
    file = open('1.jpg', 'wb')
    file.write(imgdata)
    file.close()
    # 文件写入完毕
    folder_name = "1.jpg"
    label, probability = labels_class(folder_name)
    class_name = labels_data[label[0]]
################################################################################
    if probability >= 0.7:
        user = {'accuracy': str(probability), 'is_object': 1, 'class_name': class_name}
    else:
        user = {'accuracy': str(probability), 'is_object': 0, 'class_name': class_name}
    return jsonify(results=[user])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5006)
