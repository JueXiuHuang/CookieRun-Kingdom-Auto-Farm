from PyQt5 import QtWidgets, uic, QtGui
import sys
import os
import json
import subprocess
import threading
import cv2
import numpy as np
import time
from PIL import Image
import joblib

class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi('UI.ui', self)

        self.number_clf = joblib.load('./models/number.model')

        self.config = {'Language':'zh', 'Mode':'InDepth'}

        # Try to load config file
        if os.path.exists('./config.json'):
            with open('config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        
        path = os.path.join('./images/other', self.config['Language'], 'NetworkError.png')
        self.neterror = cv2.imread(path, 0)
        path = os.path.join('./images/other', self.config['Language'], 'Confirm.png')
        self.confirm = cv2.imread(path, 0)
        
        self.StartBtn.clicked.connect(self.start_stop_running)
        self.SaveBtn.clicked.connect(self.save_config)

        self.looping = False
        self.thread = None

        # init config
        # basic ingredients
        for item in self.basic_ingredients_.children():
            if isinstance(item, QtWidgets.QCheckBox):
                name = item.objectName()
                product, config_key = name.split('_')
                product = product.replace(' ', '')
                if self.config.get(product) == None:
                    self.config[product] = {'check':False, 'min':20, 'type':'basic', 'name':product}
                else:
                    item.setChecked(self.config[product]['check'])
                item.stateChanged.connect(self.checkbox_changed)
            elif isinstance(item, QtWidgets.QLineEdit):
                name = item.objectName()
                product, config_key = name.split('_')
                product = product.replace(' ', '')
                if self.config.get(product) == None:
                    self.config[product] = {'check':False, 'min':20, 'type':'basic', 'name':product}
                else:
                    item.setText(str(self.config[product]['min']))
                item.textChanged.connect(self.min_val_changed)
                validator = QtGui.QIntValidator()
                item.setValidator(validator)
        # high end products
        for item in self.high_end_products_.children():
            if isinstance(item, QtWidgets.QCheckBox):
                name = item.objectName()
                product, config_key = name.split('_')
                product = product.replace(' ', '')
                if self.config.get(product) == None:
                    self.config[product] = {'check':False, 'min':20, 'type':'mixture'}
                else:
                    item.setChecked(self.config[product]['check'])
                item.stateChanged.connect(self.checkbox_changed)
            elif isinstance(item, QtWidgets.QLineEdit):
                name = item.objectName()
                product, config_key = name.split('_')
                product = product.replace(' ', '')
                if self.config.get(product) == None:
                    self.config[product] = {'check':False, 'min':20, 'type':'mixture'}
                else:
                    item.setText(str(self.config[product]['min']))
                item.textChanged.connect(self.min_val_changed)
                validator = QtGui.QIntValidator()
                item.setValidator(validator)

        # load template images
        self.templates = []
        lang = self.config['Language']
        for key in self.config:
            if key == 'Language' or key == 'Mode':
                continue

            fn = key + '.png'
            if self.config['Mode'] == 'InDepth':
                path = os.path.join('./images/names', lang, fn)
            else:
                pass
            img = cv2.imread(path, 0)
            self.templates.append((img, self.config[key]))

    def handle_exceptions(self, img):
        # check network error first
        score, top_left, _ = self.template_search(img, self.confirm)
        if score > 0.9:
            print('Network error')
            x, y = top_left
            self.tap(x, y)
            return True
        return False

    def main_loop(self):
        self.connect_device()
        method = cv2.TM_SQDIFF_NORMED
        next_x = 640 + 30
        next_y = 360

        while self.looping:
            img_bs = self.get_screen()
            img_bs_gray = cv2.cvtColor(img_bs, cv2.COLOR_BGR2GRAY)

            # check exceptions
            ret = self.handle_exceptions(img_bs_gray)
            if ret:
                continue

            for tmp in self.templates:
                tmp_img, tmp_config = tmp
                w, h = tmp_img.shape[1], tmp_img.shape[0]
                score, top_left, bottom_right = self.template_search(img_bs_gray, tmp_img)
                if score < 0.9 or not tmp_config['check']:
                    continue
                
                x, y = top_left
                if tmp_config['type'] == 'mixture':
                    stock = self.extract_numbers(tmp_config['type'], img_bs, x-85, y+100)
                else:
                    stock = self.extract_numbers(tmp_config['type'], img_bs)
                
                if stock >= tmp_config['min']:
                    continue

                x += w//2 + 90
                y += 140
                self.tap(x, y)
                time.sleep(0.5)
                break
            self.tap(next_x, next_y)
            time.sleep(0.5)
        return

    def predict(self, img, number_color, reverse=False):
        start = time.time()

        lower, upper = number_color

        img_hsv = cv2.cvtColor(img if not reverse else ~img, cv2.COLOR_BGR2HSV)
        img_binary = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
        img_binary = ~img_binary
        _, labels, stats, _ = cv2.connectedComponentsWithStats(img_binary, connectivity=4, ltype=None)

        # remove background component
        stats = stats[1:,:]
        
        # add label number
        stats = [(i+1,x) for i,x in enumerate(stats)]

        # check height of each connected component
        h_lower, h_upper = int(img.shape[0]*0.3), int(img.shape[0]*1)
        stats = [(idx,x) for idx,x in stats if x[3] >= h_lower and x[3] <= h_upper]

        # check center
        center = lambda x,y,w,h : (int(x+w/2), int(y+h/2))
        center_y = img.shape[0] // 2 # half of height of input image
        center_threshold = int(img.shape[0] * 0.2)
        stats = [(idx,x) for idx,x in stats if abs(center(x[0],x[1],x[2],x[3])[1]-center_y) <= center_threshold]

        # sort by x coord
        components = sorted(stats, key=lambda x : x[1][0])

        predict_size = (47, 73)
        predict_digits = np.empty((0, predict_size[0]*predict_size[1]), np.float32)
        result_img = img.copy()
        for c in components:
            idx,(x,y,w,h,_) = c
            img_digit = labels[y:y+h,x:x+w]
            img_digit = (img_digit == idx).astype(np.uint8) * 255

            img = Image.fromarray(cv2.cvtColor(img_digit, cv2.COLOR_BGR2RGB))
            img = img.convert('1')
            img = img.resize(predict_size)
            img = np.array(img, np.float32)
            img = np.reshape(img, (1, -1))
            predict_digits = np.append(predict_digits, img, axis=0)

            cv2.rectangle(result_img,(x,y),(x+w,y+h),(0,255,0),1)

        results = self.number_clf.predict(predict_digits)
        number = 0
        for r in results:
            number *= 10
            number += r

        # print(f'Cost: {time.time() - start} s')

        return number, result_img

    def extract_numbers(self, typ, img, x=None, y=None):
        if typ == 'basic':
            x = 640 + 50
            y = 25
            dx = 100
            dy = 30
            img = img[y:y+dy, x:x+dx, :]
            number, _ = self.predict(img, ([0,0,0], [180,255,130]))
        else:
            dx = 40
            dy = 24
            img = img[y:y+dy, x:x+dx, :]
            
            number, _ = self.predict(img, ([0,0,0], [180,255,130]), reverse=True)
            
        # t = time.localtime()
        # current_time = time.strftime("%H%M%S", t)
        # fn = str(number) + '_' + current_time + '.png'
        # path = os.path.join('./images/output', fn)
        # cv2.imwrite(path, img)
        print('Predict Result:', number)
        
        return number

    def sh(self, command):
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = p.stdout.read()
        return result

    def get_screen(self):
        command = 'adb -s 127.0.0.1:5555 shell screencap -p'
        result = self.sh(command)
        image_bytes = result.replace(b'\r\n', b'\n')
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return image

    def tap(self, x, y):
        command = 'adb -s 127.0.0.1:5555 shell input tap ' + str(x) + ' ' + str(y)
        self.sh(command)

    def connect_device(self):
        command = 'adb connect 127.0.0.1:5555'
        self.sh(command)

    def template_search(self, img_src, img_tmp):
        method = cv2.TM_CCOEFF_NORMED
        w, h = img_tmp.shape[1], img_tmp.shape[0]

        result = cv2.matchTemplate(img_src, img_tmp, method)
        min_val, max_val, min_loc, top_left = cv2.minMaxLoc(result)
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return max_val, top_left, bottom_right

    def checkbox_changed(self):
        sender = self.sender()
        name, _ = sender.objectName().split('_')
        state = sender.isChecked()
        self.config[name]['check'] = state
        print(name, state)
        return

    def min_val_changed(self):
        sender = self.sender()
        name, _ = sender.objectName().split('_')
        val = sender.text()
        self.config[name]['min'] = int(val)
        print(name, val)
        return

    def start_stop_running(self):
        if not self.looping:
            if self.thread is None or not self.thread.is_alive():
                self.looping = True
                self.StartBtn.setText('Stop')

                self.thread = threading.Thread(target = self.main_loop)
                self.thread.setDaemon(True)
                self.thread.start()
        else:
            self.looping = False
            self.StartBtn.setText('Start')
        return

    def save_config(self):
        print('Save config file')
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)
        return

app = QtWidgets.QApplication(sys.argv)
window = UI()
window.show()

app.exec_()