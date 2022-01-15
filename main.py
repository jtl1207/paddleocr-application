import cv2, difflib, re, sys, time, snap7, threading, os, Img, subprocess

import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from paddleocr import PaddleOCR, draw_ocr
from style import Ui_mainWindow

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_mainWindow()  # 前端布局文件导入
        self.ui.setupUi(self)

        # 共享元素
        self.ppocr = None  # 模型
        self.show = None  # cv2预处理的图像
        self.showend = None  # 绘制完成输出图像
        self.midimg = None  # 未标记的输出图像
        self.testtarget = ""  # 输入的检测目标
        self.historylist = []  # 历史记录
        self.save = ""  # 需要保存的结果
        self.com_out_ans_bool = 0  # 检测输出的结果
        self.thetime = None  # 显示的时间
        self.hashmark = [0, 0, 0, 0, 0, 0, 0]
        self.hashans = [0, 0, 0, 0, 0, 0, 0]
        self.bad_str = ''
        # 状态变量
        self.readly_check = False  # 准备检测
        self.checking = False  # 检测中
        self.need_check = False  # 需要检测
        self.need_clean = False  # 报警清除
        self.need_save_bool = False  # 需要保存状态位
        self.need_freshen = False  # 需要更新输出
        self.need_com_freshen = False  # hmi输出更新
        self.need_learn = False  # 需要重新学习模板
        # 参数变量
        self.img_colour_mod = 1  # 图像模式
        self.img_mod = []
        self.img_blus_mod = '0'  # 滤波模式
        self.img_colour2_mod = 0  # 彩图处理
        self.img_gray_mod = 0  # 直方图处理模式
        self.usecn = 0  # 使用中文
        self.usechar = 0  # 使用符号
        self.opentime = 0  # 腐蚀次数
        self.closetime = 0  # 膨胀次数
        self.jdtime = 3  # 均值精度
        self.lbtime = 5  # 均值参数
        self.jd2time = 3  # 颜色提取精度
        self.lb2time = 5  # 颜色提取参数
        self.maxbad = 0  # 容错率
        self.maxmark = 95  # 容错率
        self.com_out_bad = 0  # 输出的不良字符数
        self.bad_mark = 0  # 不良分值
        # 通讯变量
        self.com_out = 0  # 通讯输出
        '''vb5000>vb5001
        Q0.0    1.电源
        Q0.1    2.准备检测
        Q0.2    4.严重报警
        Q0.3    8.结果1:合格
        Q0.4    16.结果2:不良
        '''
        self.com_in = None  # 通讯值
        self.com_in_en = False
        '''vb5002
        I0.0    1.使能
        I0.1    2.触发检测
        I0.2    4.触发学习(未使用)
        I0.3    8.设备报警清除
        '''
        self.badnum = [1, 1, 0, 1, 0, 0, 0, 0]  # 报错代码
        self.com_bad = 0
        '''vb5003
                0.无错误
        Q1.0    1.模型准备中
        Q1.1    2.摄像头连接中断(硬件)
        Q1.2    4.
        Q1.3    8.通讯断开
        Q1.4    16.短期内不良瓶数过多警告
        Q1.5    32.未设置检测目标基准
        Q1.6    64.
        Q1.7    128.
        '''
        self.com_out_time = 0  # 本次检测时间ms
        self.com_out_optimize = ''  # plc等待时间
        '''vw5004
        int
        '''
        self.com_out_ans_len = 0  # 字符串长度
        '''vw5006
        vw5005:int 字符串长度
        '''

        self.com_out_fullans = ''  # 字符串
        self.com_out_ans = ''  # 不带换行与中文的字符串
        '''vb5008-vb5108
        string
        vb5008-vb5108:ascii字符(无中文)
        '''
        self.com_out_mark = 0  # 置信度
        # 输出颜色
        self.colour_en = False
        self.colour_check = False
        self.colour_clean = False
        self.colour_readly = False
        self.colour_bad = False
        self.colour_str = False
        # 初始化
        self.doit()  # 设置按钮事件
        self.startset()  # 读取存档并设置参数

        self.time_mod = threading.Thread(target=self.mod_start)  # 模型初始化
        self.time_mod.setDaemon(True)
        self.time_mod.start()

        self.cap = cv2.VideoCapture()
        # 画面宽度设定为 640
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # 画面高度度设定为 480
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.open(0)
        self.time_com_camera = QtCore.QTimer(self)
        self.time_com_camera.timeout.connect(self.show_camera)  # 图像读取
        self.time_com_camera.start(30)

        self.plc = snap7.client.Client()  # plc通讯
        self.plc.set_connection_type(3)
        self.time_com = threading.Thread(target=self.plc_com)  # 通讯线程
        self.time_com.setDaemon(True)
        self.time_com.start()

        self.time_com_iq = QtCore.QTimer(self)
        self.time_com_iq.timeout.connect(self.iq_count)  # 界面数据刷新
        self.time_com_iq.start(20)

        self.time_main = threading.Thread(target=self.plcdo)  # 副线程
        self.time_main.setDaemon(True)
        self.time_main.start()

        self.time_clock = threading.Thread(target=self.clock)  # 时钟线程
        self.time_clock.setDaemon(True)
        self.time_clock.start()


    def mod_start(self):
        """
        模型初始化
        """
        self.ppocr = PaddleOCR(
            use_gpu=True,  # 使用gpu
            cls=True,  # 角度分类
            det_limit_side_len=320,  # 检测算法前向时图片长边的最大尺寸，
            det_limit_type='max',  # 限制输入图片的大小,可选参数为limit_type[max, min] 一般设置为 32 的倍数，如 960。
            ir_optim=False,
            use_fp16=False,  # 16位半精度
            use_tensorrt=False,  # 使用张量
            gpu_mem=6000,  # 初始化占用的GPU内存大小
            cpu_threads=20,
            enable_mkldnn=True,  # 是否启用mkldnn
            max_batch_size=512,  # 图片尺寸最大大小
            cls_model_dir='data/mod/2.3.0.1/ocr/cls/ch_ppocr_mobile_v2.0_cls_infer',
            # cls模型位置
            # image_dir="",  # 通过命令行调用时间执行预测的图片或文件夹路径
            det_algorithm='DB',  # 使用的检测算法类型DB/EAST
            det_model_dir='data/mod/2.3.0.1/ocr/det/ch/ch_PP-OCRv2_det_infer',
            # 检测模型所在文件夹。传参方式有两种，1. None: 自动下载内置模型到 ~/.paddleocr/det；2.自己转换好的inference模型路径，模型路径下必须包含model和params文件
            # DB(还有east,SAST)
            det_db_thresh=0.3,  # DB模型输出预测图的二值化阈值
            det_db_box_thresh=0.6,  # DB模型输出框的阈值，低于此值的预测框会被丢弃
            det_db_unclip_ratio=1.5,  # DB模型输出框扩大的比例
            use_dilation=True,  #缩放图片
            det_db_score_mode="fast",  #计算分数模式,fast对应原始的rectangle方式，slow对应polygon方式。
            # 文本识别器的参数
            rec_algorithm='CRNN',  # 使用的识别算法类型
            rec_model_dir='data/mod/2.3.0.1/ocr/rec/ch/ch_PP-OCRv2_rec_infer',
            # 识别模型所在文件夹。传承那方式有两种，1. None: 自动下载内置模型到 ~/.paddleocr/rec；2.自己转换好的inference模型路径，模型路径下必须包含model和params文件
            rec_image_shape="3,32,320",  # 识别算法的输入图片尺寸
            cls_batch_num=36,  #
            cls_thresh=0.9,  #
            lang='ch',  # 语言
            det=True,  # 检测文字位置
            rec=True,  # 识别文字内容
            use_angle_cls=False,  # 识别竖排文字
            rec_batch_num=36,  # 进行识别时，同时前向的图片数
            max_text_length=25,  # 识别算法能识别的最大文字长度
            rec_char_dict_path='data/mod/2.3.0.1/ppocr_keys_v1.txt',  # 识别模型字典路径，当rec_model_dir使用方式2传参时需要修改为自己的字典路径
            use_space_char=True,  # 是否识别空格
        )
        self.badnum[0] = 0

    def plc_com(self):
        """
        通讯线程
        """
        while 1:
            if self.plc.get_connected():
                try:
                    self.badnum[3] = 0
                    # 输出
                    self.plc.write_area(snap7.types.Areas.DB, 1, 5000, bytearray([self.com_out]))
                    self.plc.write_area(snap7.types.Areas.DB, 1, 5003, bytearray([self.com_bad]))
                    # 读取输入的信号
                    self.com_in = self.plc.read_area(snap7.types.Areas.DB, 1, 5002, 1)
                    comin = self.com_in[0]
                    if comin % 2 == 1:
                        self.com_in_en = True
                        comin -= 1
                        if comin >= 8:
                            self.need_clean = True
                            comin -= 8
                        if comin == 2 and self.checking == False:
                            self.need_check = True
                    else:
                        self.com_in_en = False
                    self.plc.write_area(snap7.types.Areas.DB, 1, 5002, bytearray(b'\x00'))
                    # 刷新输出结果
                    if self.need_com_freshen:
                        dw = bytearray()
                        le = bytearray()
                        snap7.util.set_int(dw, 256, self.com_out_time)
                        self.plc.write_area(snap7.types.Areas.DB, 1, 5004, dw)
                        snap7.util.set_int(le, 256, self.com_out_ans_len)
                        self.plc.write_area(snap7.types.Areas.DB, 1, 5006, le)
                        x = re.sub("[年]", "(N)", self.com_out_ans)
                        x = re.sub("[月]", "(Y)", x)
                        x = re.sub("[日]", "(R)", x)
                        data = bytearray(100)
                        snap7.util.set_string(data, 0, x, 255)
                        self.plc.write_area(snap7.types.Areas.DB, 1, 5008, data)
                        self.freshen = False
                except:
                    self.plc.disconnect()
                    self.com_in_en = False
                    self.badnum[3] = 1
            else:
                self.com_in_en = False
                try:
                    self.plc.connect("192.168.2.1", 0, 1)
                except:
                    self.badnum[3] = 1
            time.sleep(0.0001)

    def iq_count(self):
        """
        信号统计
        """
        # 准备检测
        if self.badnum[0:2] == [0, 0] and self.checking == False:
            self.readly_check = True
        else:
            self.readly_check = False
        # 报警清除
        if self.need_clean:
            self.bad_mark = 0
            self.badnum[4] = 0
            self.need_clean = False
            self.com_out_optimize = '无'
        # 报警统计
        if self.testtarget == '':
            self.badnum[5] = 1
        else:
            self.badnum[5] = 0
        if self.bad_mark > 3:
            self.badnum[4] = 1
        else:
            self.badnum[4] = 0
        # 输出错误码
        j = 0
        for i in range(7):
            if self.badnum[i] == 1:
                j += 2 ** i
        self.com_bad = j
        # 输出的信号
        q = 1
        if self.readly_check: q += 2
        if self.badnum[0:3] != [0, 0, 0]: q += 4
        if self.com_out_ans_bool == 1:
            q += 8
        elif self.com_out_ans_bool == 2:
            q += 16
        self.com_out = q
        # 刷新界面
        if self.readly_check != self.colour_readly:
            self.colour_readly = self.readly_check
            if self.readly_check:
                self.ui.label__com_out_readly.setStyleSheet('border-radius: 10px; background-color: rgb(0, 250, 0);')
            else:
                self.ui.label__com_out_readly.setStyleSheet('border-radius: 10px; background-color: red;')

        if (self.badnum[0:4] != [0, 0, 0, 0]) != self.colour_bad:
            self.colour_bad = self.badnum[0:4] != [0, 0, 0, 0]
            if self.colour_bad:
                self.ui.label_com_out_bad.setStyleSheet('border-radius: 10px; background-color: rgb(0, 250, 0);')
            else:
                self.ui.label_com_out_bad.setStyleSheet('border-radius: 10px; background-color: red;')
        if self.com_in_en != self.colour_en:
            self.colour_en = self.com_in_en
            if self.com_in_en:
                self.ui.label_com_in_en.setStyleSheet('border-radius: 10px; background-color: rgb(0, 250, 0);')
            else:
                self.ui.label_com_in_en.setStyleSheet('border-radius: 10px; background-color: red;')
        if self.need_check != self.colour_check:
            self.colour_check = self.need_check
            if self.need_check:
                self.ui.label_com_in_do.setStyleSheet('border-radius: 10px; background-color: rgb(0, 250, 0);')
            else:
                self.ui.label_com_in_do.setStyleSheet('border-radius: 10px; background-color: red;')
        if self.need_clean != self.colour_clean:
            self.colour_clean = self.need_clean
            if self.need_clean:
                self.ui.label_com_in_clean.setStyleSheet('border-radius: 10px; background-color: rgb(0, 250, 0);')
            else:
                self.ui.label_com_in_clean.setStyleSheet('border-radius: 10px; background-color: red;')
        srt = ''
        if self.badnum[0:6] == [0, 0, 0, 0, 0, 0]:
            colour = True
            srt = '设备运行正常'
        else:
            colour = False
            if self.badnum[0] == 1:
                srt += '等待模型初始化\n'
            if self.badnum[1] == 1:
                srt += '摄像头连接中断\n'
            if self.badnum[3] == 1:
                srt += 'PLC通讯断开\n'
            if self.badnum[4] == 1:
                srt += '短期内不良数过多\n'
            if self.badnum[5] == 1:
                srt += '未设置检测目标基准\n'
                self.need_learn = True
        self.ui.label_bad_str.setText(srt)
        if self.colour_str != colour:
            self.colour_str = colour
            if self.colour_str:
                self.ui.label_bad_str.setStyleSheet('color: rgb(0, 200, 0);')
            else:
                self.ui.label_bad_str.setStyleSheet('color: rgb(255, 0, 0);')
        # 刷新输出
        if self.need_freshen:
            self.freshen_interface()
            self.need_freshen = False

    def clock(self):
        """
        时钟
        """
        while 1:
            self.thetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.ui.label_time.setText(str(self.thetime))
            time.sleep(1)

    def show_camera(self):
        """
        摄像头图像传输(线程1)
        """
        if self.cap.isOpened() == True:
            self.badnum[1] = 0
            flag, img0 = self.cap.read()
            img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            # 这里是正常图像
            showImage = QtGui.QImage(img1.data, img1.shape[1], img1.shape[0], QtGui.QImage.Format.Format_RGB888)
            self.show = img0
            self.ui.label_11.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.badnum[1] = 1

    def doit(self):
        """
        按钮槽函数写入
        """
        self.ui.butmain.clicked.connect(self.butmain)
        self.ui.buthelp.clicked.connect(self.buthelp)
        self.ui.but.clicked.connect(self.butdo)
        self.ui.but_1.clicked.connect(self.tab_1)
        self.ui.but_2.clicked.connect(self.tab_2)
        self.ui.but_3.clicked.connect(self.butclean)
        # 虚拟键盘
        self.ui.Button0.clicked.connect(self.button0)
        self.ui.Button1.clicked.connect(self.button1)
        self.ui.Button2.clicked.connect(self.button2)
        self.ui.Button3.clicked.connect(self.button3)
        self.ui.Button4.clicked.connect(self.button4)
        self.ui.Button5.clicked.connect(self.button5)
        self.ui.Button6.clicked.connect(self.button6)
        self.ui.Button7.clicked.connect(self.button7)
        self.ui.Button8.clicked.connect(self.button8)
        self.ui.Button9.clicked.connect(self.button9)
        self.ui.Buttonback.clicked.connect(self.buttonback)
        self.ui.Buttonc.clicked.connect(self.buttonc)
        self.ui.Buttond.clicked.connect(self.buttond)
        self.ui.Buttondd.clicked.connect(self.buttondd)

        self.ui.Buttonent.clicked.connect(self.buttonent)
        self.ui.Buttoni.clicked.connect(self.buttoni)
        self.ui.Buttonl.clicked.connect(self.buttonl)

        self.ui.Buttonyear.clicked.connect(self.buttoyear)
        self.ui.Buttonmonth.clicked.connect(self.buttonmonth)
        self.ui.Buttonday.clicked.connect(self.buttonday)

        self.ui.pushButton_4.clicked.connect(self.add)
        self.ui.pushButton_5.clicked.connect(self.dec)

        self.ui.radioButtongray.clicked.connect(self.radioButtongray_check)
        self.ui.radioButtonmould.clicked.connect(self.radioButtonmould_check)
        self.ui.radioButtonblack.clicked.connect(self.radioButtonblack_check)
        self.ui.radioButtoncolor.clicked.connect(self.radioButtoncolor_check)

        self.ui.radioButton_colorhis.clicked.connect(self.radioButton_colorhis_check)
        self.ui.radioButtongauss.clicked.connect(self.radioButton_check_end)
        self.ui.radioButtoneven.clicked.connect(self.radioButton_check_end)
        self.ui.radioButton_his.clicked.connect(self.radioButton_his_check)
        self.ui.radioButton_hisauto.clicked.connect(self.radioButton_hisauto_check)
        self.ui.radioButton_hsv.clicked.connect(self.radioButton_hsv_check)
        self.ui.radioButton_hisall.clicked.connect(self.radioButton_hisall_check)
        self.ui.Buttoncn.clicked.connect(self.Buttoncn_check)
        self.ui.Buttoncn_2.clicked.connect(self.Buttoncn_2_check)
        self.ui.radioButtonopen.clicked.connect(self.radioButton_check_end)
        self.ui.radioButtonclose.clicked.connect(self.radioButton_check_end)
        self.ui.pc_close.clicked.connect(self.pc_close)
        self.ui.pushButton_close1.clicked.connect(self.pc_close1)
        self.ui.pushButton_close2.clicked.connect(self.pc_close2)
        self.ui.pushButton_open1.clicked.connect(self.pc_open1)
        self.ui.pushButton_open2.clicked.connect(self.pc_open2)
        self.ui.radioButton_otus.clicked.connect(self.butblack)
        self.ui.radioButton_200.clicked.connect(self.butblack)
        self.ui.radioButton_mean.clicked.connect(self.butblack)
        self.ui.pushButton_6.clicked.connect(self.submaxmark)
        self.ui.pushButton_7.clicked.connect(self.addmaxmark)
        self.ui.buthelp_2.clicked.connect(self.help_2)
        self.ui.pushButton_jd1.clicked.connect(self.subjd)
        self.ui.pushButton_jd2.clicked.connect(self.addjd)
        self.ui.pushButton_lb1.clicked.connect(self.sublb)
        self.ui.pushButton_lb2.clicked.connect(self.addlb)
        self.ui.pushButton_jd1_2.clicked.connect(self.subjd_2)
        self.ui.pushButton_jd2_2.clicked.connect(self.addjd_2)
        self.ui.pushButton_lb1_2.clicked.connect(self.sublb_2)
        self.ui.pushButton_lb2_2.clicked.connect(self.addlb_2)
        self.ui.pushButtoncleanlearn.clicked.connect(self.mod_clean)

    def radioButton_colorhis_check(self):
        if self.ui.radioButton_colorhis.isChecked():
            self.img_colour2_mod = 1
        else:
            self.img_colour2_mod = 0
        self.need_save_bool = True

    def mod_clean(self):
        self.need_learn = True

    def subjd(self):
        if self.jdtime > 3:
            self.jdtime -= 2
            self.ui.label_jd.setText(str(self.jdtime))
            self.need_save_bool = True
            self.need_learn = True

    def addjd(self):
        if self.jdtime < 60:
            self.jdtime += 2
            self.ui.label_jd.setText(str(self.jdtime))
            self.need_save_bool = True
            self.need_learn = True

    def sublb(self):
        if self.lbtime > 1:
            self.lbtime -= 1
            self.ui.label_lb.setText(str(self.lbtime))
            self.need_save_bool = True
            self.need_learn = True

    def addlb(self):
        if self.lbtime < 20:
            self.lbtime += 1
            self.ui.label_lb.setText(str(self.lbtime))
            self.need_save_bool = True
            self.need_learn = True

    def subjd_2(self):
        if self.jd2time > 1:
            self.jd2time -= 1
            self.ui.label_jd_2.setText(str(self.jd2time))
            self.need_save_bool = True
            self.need_learn = True

    def addjd_2(self):
        if self.jd2time < 10:
            self.jd2time += 1
            self.ui.label_jd_2.setText(str(self.jd2time))
            self.need_save_bool = True
            self.need_learn = True

    def sublb_2(self):
        if self.lb2time > 1:
            self.lb2time -= 1
            self.ui.label_lb_2.setText(str(self.lb2time))
            self.need_save_bool = True
            self.need_learn = True

    def addlb_2(self):
        if self.lb2time < 20:
            self.lb2time += 1
            self.ui.label_lb_2.setText(str(self.lb2time))
            self.need_save_bool = True
            self.need_learn = True

    def help_2(self):
        self.ui.tabWidget.setCurrentIndex(2)

    def addmaxmark(self):
        if self.maxmark < 100:
            self.maxmark += 1
            self.ui.label_bad_4.setText(f'{self.maxmark}')
            self.need_save_bool = True

    def submaxmark(self):
        if self.maxmark > 60:
            self.maxmark -= 1
            self.ui.label_bad_4.setText(f'{self.maxmark}')
            self.need_save_bool = True

    def butblack(self):
        if self.ui.radioButton_otus.isChecked():
            self.img_black_mod = 1
            self.ui.widget_7.hide()
            self.ui.widget_8.hide()
        elif self.ui.radioButton_mean.isChecked():
            self.img_black_mod = 2
            self.ui.widget_7.show()
            self.ui.widget_8.hide()
        elif self.ui.radioButton_200.isChecked():
            self.img_black_mod = 3
            self.ui.widget_8.show()
            self.ui.widget_7.hide()
        self.need_save_bool = True
        self.need_learn = True

    def butdo(self):
        self.do()

    def butclean(self):
        self.need_clean = True

    def button0(self):
        self.testtarget = self.testtarget + "0"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button1(self):
        self.testtarget += "1"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button2(self):
        self.testtarget += "2"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button3(self):
        self.testtarget += "3"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button4(self):
        self.testtarget += "4"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button5(self):
        self.testtarget += "5"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button6(self):
        self.testtarget += "6"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button7(self):
        self.testtarget += "7"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button8(self):
        self.testtarget += "8"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def button9(self):
        self.testtarget += "9"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttonback(self):
        self.testtarget = self.testtarget[:-1]
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttonc(self):
        self.testtarget = ""
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttond(self):
        self.testtarget += "."
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttondd(self):
        self.testtarget += ":"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttonent(self):
        self.testtarget += "\n"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_learn = True

    def buttoni(self):
        self.testtarget += "-"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttonl(self):
        self.testtarget += "/"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttoyear(self):
        self.testtarget += "年"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttonmonth(self):
        self.testtarget += "月"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    def buttonday(self):
        self.testtarget += "日"
        self.ui.inin.setPlainText(self.testtarget)
        self.need_save_bool = True
        self.need_learn = True

    # 色彩模式按钮
    def radioButtoncolor_check(self):
        self.img_colour_mod = 4
        self.ui.widget_6.show()
        self.ui.widget_5.hide()
        self.ui.widget_3.hide()
        self.ui.widget_4.hide()
        self.ui.pushButtoncleanlearn.hide()
        self.radioButtonblack_check_end()
        self.need_save_bool = True

    def radioButtongray_check(self):
        self.img_colour_mod = 1
        self.ui.widget_3.show()
        self.ui.widget_5.hide()
        self.ui.widget_6.hide()
        self.ui.pushButtoncleanlearn.hide()
        self.radioButtonblack_check_end()
        if self.img_gray_mod == 0:
            self.ui.widget_4.hide()
        else:
            self.ui.widget_4.show()
        self.need_save_bool = True

    def radioButtonmould_check(self):
        self.img_colour_mod = 3
        self.ui.widget_4.show()
        self.ui.widget_3.hide()
        self.ui.widget_6.hide()
        self.ui.widget_5.show()
        self.ui.pushButtoncleanlearn.show()
        self.radioButtonblack_check_end()
        self.need_save_bool = True

    def radioButtonblack_check(self):
        self.img_colour_mod = 2
        self.ui.widget_4.show()
        self.ui.widget_3.hide()
        self.ui.widget_6.hide()
        self.ui.widget_5.show()
        self.ui.pushButtoncleanlearn.hide()
        self.radioButtonblack_check_end()
        self.need_save_bool = True

    def radioButtonblack_check_end(self):
        if self.img_colour_mod != 1:
            self.ui.radioButtongray.setChecked(False)
        if self.img_colour_mod != 3:
            self.ui.radioButtonmould.setChecked(False)
        if self.img_colour_mod != 2:
            self.ui.radioButtonblack.setChecked(False)
        if self.img_colour_mod != 4:
            self.ui.radioButtoncolor.setChecked(False)
        if self.img_black_mod == 1:
            self.ui.widget_7.hide()
            self.ui.widget_8.hide()
        elif self.img_black_mod == 2:
            self.ui.widget_7.show()
            self.ui.widget_8.hide()
        elif self.img_black_mod == 3:
            self.ui.widget_8.show()
            self.ui.widget_7.hide()

    # 直方图按钮
    def radioButton_his_check(self):
        self.img_gray_mod = 1
        self.radioButtonhis_check_end()

    def radioButton_hsv_check(self):
        self.img_gray_mod = 4
        self.radioButtonhis_check_end()

    def radioButton_hisauto_check(self):
        self.img_gray_mod = 2
        self.radioButtonhis_check_end()

    def radioButton_hisall_check(self):
        self.img_gray_mod = 3
        self.radioButtonhis_check_end()

    def radioButtonhis_check_end(self):
        """
        直方图数据处理
        """
        self.ui.radioButton_hisauto.setChecked(False)
        self.ui.radioButton_hisall.setChecked(False)
        self.ui.radioButton_his.setChecked(False)
        self.ui.radioButton_hsv.setChecked(False)
        if self.img_gray_mod == 1:
            self.ui.radioButton_his.setChecked(True)
        elif self.img_gray_mod == 2:
            self.ui.radioButton_hisauto.setChecked(True)
        elif self.img_gray_mod == 3:
            self.ui.radioButton_hisall.setChecked(True)
        elif self.img_gray_mod == 4:
            self.ui.radioButton_hsv.setChecked(True)
        if self.img_gray_mod == 0:
            self.ui.widget_4.hide()
        else:
            self.ui.widget_4.show()
        self.need_save_bool = True
        self.need_learn = True

    # 滤波按钮
    def radioButton_check_end(self):
        self.img_blus_mod = '0'
        if self.ui.radioButtongauss.isChecked():
            self.img_blus_mod += '1'
        if self.ui.radioButtoneven.isChecked():
            self.img_blus_mod += '2'
        if self.ui.radioButtonopen.isChecked():
            self.img_blus_mod += '3'
        if self.ui.radioButtonclose.isChecked():
            self.img_blus_mod += '4'
        self.need_save_bool = True

    def pc_open1(self):
        if self.opentime > 0:
            self.opentime -= 1
            self.ui.label_open.setText(str(self.opentime))
            self.need_save_bool = True
            self.need_learn = True

    def pc_open2(self):
        if self.opentime < 5:
            self.opentime += 1
            self.ui.label_open.setText(str(self.opentime))
            self.need_save_bool = True
            self.need_learn = True

    def pc_close1(self):
        if self.closetime > 0:
            self.closetime -= 1
            self.ui.label_close.setText(str(self.closetime))
            self.need_save_bool = True
            self.need_learn = True

    def pc_close2(self):
        if self.closetime < 5:
            self.closetime += 1
            self.ui.label_close.setText(str(self.closetime))
            self.need_save_bool = True
            self.need_learn = True

    def pc_close(self):
        """
        关机
        """
        self.scvechange()
        # 没有测试过的代码
        # path = os.getcwd()
        # cmd1 = f'C:\Windows\System32\schtasks /create /tn "My App" /tr {path}\开始程序.exe /sc onlogon'
        # cmd2 = 'C:\Windows\System32\schtasks  /Query  /tn "My App"'
        # p = subprocess.run(cmd2, capture_output=True, shell=True, encoding="gbk")
        # if len(p.stderr)!=0:
        #     p = subprocess.run(cmd1, capture_output=True, shell=True, encoding="gbk")
        subprocess.run('C:\Windows\System32\shutdown -s -t 0')
        sys.exit()

    # 容错率调整
    def add(self):
        if self.maxbad < 10:
            self.maxbad += 1
            self.ui.label_bad.setText(str(self.maxbad))
        self.need_save_bool = True

    def dec(self):
        if self.maxbad > 0:
            self.maxbad -= 1
            self.ui.label_bad.setText(str(self.maxbad))
        self.need_save_bool = True

    def Buttoncn_check(self):
        if self.usecn == 1:
            self.usecn = 0
        else:
            self.usecn = 1
        if self.usecn == 1:
            self.ui.Buttoncn.setText('禁用\n中文')
        else:
            self.ui.Buttoncn.setText('使用\n中文')
        self.need_save_bool = True
        self.need_learn = True

    def Buttoncn_2_check(self):

        if self.usechar == 1:
            self.usechar = 0
        else:
            self.usechar = 1
        if self.usechar == 1:
            self.ui.Buttoncn_2.setText('禁用\n符号')
        else:
            self.ui.Buttoncn_2.setText('使用\n符号')
        self.need_save_bool = True
        self.need_learn = True

    # tab切换
    def butmain(self):
        self.ui.tabWidget.setCurrentIndex(0)

    # tab切换
    def buthelp(self):
        self.ui.tabWidget.setCurrentIndex(1)

    # 滤波
    def blur(self, img):
        kernel = np.ones((3, 3), np.uint8)
        if re.sub("[^2]", "", self.img_blus_mod) == "2":
            img = cv2.blur(img, (3, 3), 0)
        if re.sub("[^1]", "", self.img_blus_mod) == "1":
            img = cv2.GaussianBlur(img, (3, 3), 0)
        if re.sub("[^3]", "", self.img_blus_mod) == "3":
            img = cv2.erode(img, kernel, iterations=self.opentime)
            img = cv2.dilate(img, kernel, iterations=self.closetime)
        if re.sub("[^4]", "", self.img_blus_mod) == "4":
            img = cv2.dilate(img, kernel, iterations=self.closetime)
            img = cv2.erode(img, kernel, iterations=self.opentime)
        return img

    def plcdo(self):
        while 1:
            if self.checking != True and self.need_check:
                self.need_check = False
                self.checking = True
                t = time.time()
                if self.need_save_bool:
                    self.scvechange()
                    if self.img_colour_mod == 3:
                        self.match_learn()
                    else:
                        self.ocr()
                else:
                    if self.img_colour_mod == 3:
                        self.match()
                    else:
                        self.ocr()
                self.com_out_time = int((time.time() - t) * 1000)  # 检测时间
                self.need_freshen = True
                self.need_com_freshen = True
                self.checking = False
            time.sleep(0.0001)

    # 检测触发
    def do(self):
        if self.checking != True:
            t = time.time()
            self.need_check = False
            self.checking = True
            if self.need_save_bool:
                self.scvechange()

            if self.img_colour_mod == 3:
                if self.need_learn:
                    self.match_learn()
                else:
                    self.match()
            else:
                self.ocr()
            self.com_out_time = int((time.time() - t) * 1000)  # 检测时间
            self.need_freshen = True
            self.need_com_freshen = True
            self.checking = False

    # 模板学习
    def match_learn(self):
        if self.testtarget == '':
            self.ui.radioButtonmould.setChecked(False)
            self.ocr()
            self.com_out_ans_bool = 4
            self.com_out_bad = 0
            self.save = f'{str(time.strftime("%H:%M:%S", time.localtime()))}   学习失败  未设置检测目标\n'
            self.com_out_fullans = '模板匹配模式'
            self.com_out_optimize = '请设置检测目标'
        else:
            t0 = time.time()
            self.midimg = self.img_processing()
            result = self.ppocr.ocr(self.midimg, det=True, rec=True, cls=False)
            text, real = self.data_processing(result)
            self.com_out_mark = int(real * 100)  # 输出的置信度
            self.save = str(time.strftime("%H:%M:%S", time.localtime()))  # 历史记录
            z = self.testtarget.split()  # 识别对比参数
            outplc = ''
            outpc = ''
            outlen = 0
            znum = 0
            bada = 0
            badb = 0
            result = {}
            if len(text) == len(z):
                for line in text:
                    outplc += f'{line[1]} '
                    outlen += len(line[1])
                    outpc += f'{line[1]}\n'
                    if line[1] != z[znum]:
                        for x in line[1]:
                            result[x] = result.get(x, 0) + 1
                        for x in z[znum]:
                            result[x] = result.get(x, 0) - 1
                    znum += 1
                for x in result:
                    if result[x] < 0:
                        bada -= result[x]
                    if result[x] > 0:
                        badb += result[x]
                if bada < badb: bada = badb
                if bada == 0 and self.com_out_mark >= 95:
                    self.com_out_ans_bool = 3
                    self.com_out_bad = bada
                    self.cut_img(text)
                    self.save += f'   学习完成  学习时间:{(time.time() - t0) * 1000:.0f}ms  置信度:{self.com_out_mark}%\n'
                    self.com_out_fullans = outpc
                    self.com_out_ans = outplc
                    self.com_out_ans_len = outlen
                else:
                    self.com_out_ans_bool = 4
                    self.com_out_bad = bada
                    self.cut_img(text)
                    self.save += f'   学习失败  学习时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                    self.com_out_fullans = outpc
                    self.com_out_ans = outplc
                    self.com_out_ans_len = outlen
            else:
                if len(text) < len(z):
                    for line in z:
                        for x in line:
                            result[x] = result.get(x, 0) - 1
                    for line in text:
                        outplc += f'{line[1]} '
                        outlen += len(line[1])
                        outpc += f'{line[1]}\n'
                        for x in line[1]:
                            result[x] = result.get(x, 0) + 1
                    for x in result:
                        if result[x] < 0:
                            bada -= result[x]
                        else:
                            badb += result[x]
                    if bada < badb: bada = badb
                    if bada == 0 and self.com_out_mark >= 95:
                        self.com_out_ans_bool = 3
                        self.com_out_bad = bada
                        self.cut_img(text)
                        self.save += f'   学习成功  学习时间:{(time.time() - t0) * 1000:.0f}ms  置信度:{self.com_out_mark}%\n'
                        self.com_out_fullans = outpc
                        self.com_out_ans = outplc
                        self.com_out_ans_len = outlen
                    else:
                        self.com_out_ans_bool = 4
                        self.com_out_bad = bada
                        self.cut_img(text)
                        self.save += f'   学习失败  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                        self.com_out_fullans = outpc
                        self.com_out_ans = outplc
                        self.com_out_ans_len = outlen
                if len(text) > len(z):
                    text2 = text.copy()
                    for line in z:
                        i = 0
                        stri = ""
                        numi = 0
                        num = 0
                        for line2 in text2:
                            s = difflib.SequenceMatcher(None, line, line2[1]).ratio()
                            if s >= i:
                                i = s
                                stri = line2
                                num = numi
                            numi += 1
                        if i == 1.0:
                            del text2[num]
                        else:
                            for x in line:
                                result[x] = result.get(x, 0) + 1
                            for x in stri[1]:
                                result[x] = result.get(x, 0) - 1
                            del text2[num]
                    for list in text2:
                        l = list[1]
                        m = 0
                        for list2 in text:
                            if l == list2[1]:
                                del text[m]
                            m += 1
                    mark = 0
                    for list in text:
                        outplc += f'{list[1]} '
                        outlen += len(list[1])
                        outpc += f'{list[1]}\n'
                        mark += list[2]
                    if len(text) != 0:
                        mark = mark / len(text)
                    else:
                        mark = 0
                    for x in result:
                        if result[x] < 0:
                            bada -= result[x]
                        if result[x] > 0:
                            badb += result[x]
                    self.com_out_mark = int(mark * 100)  # 输出的置信度
                    if bada < badb: bada = badb
                    if bada == 0 and self.com_out_mark >= 85:
                        self.com_out_ans_bool = 3
                        self.com_out_bad = bada
                        self.cut_img(text)
                        self.save += f'   学习成功  学习时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                        self.com_out_fullans = outpc
                        self.com_out_ans = outplc
                        self.com_out_ans_len = outlen
                    else:
                        self.com_out_ans_bool = 4
                        self.com_out_bad = bada
                        self.cut_img(text)
                        self.save += f'   学习失败  学习时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                        self.com_out_fullans = outpc
                        self.com_out_ans = outplc
                        self.com_out_ans_len = outlen
            boxes = [line[0] for line in text]
            endimg = draw_ocr(self.midimg, boxes)
            showend = QtGui.QImage(endimg.data, endimg.shape[1], endimg.shape[0],
                                   QtGui.QImage.Format.Format_RGB888)
            self.ui.label_22.setPixmap(QtGui.QPixmap.fromImage(showend))

    def cut_img(self, text):
        if self.com_out_ans_bool == 3:
            self.com_out_optimize = '无'
            self.need_learn = False
            if self.img_colour_mod == 3:
                self.img_mod = []
                num = 0
                for line in text:
                    img = Img.cut(self.midimg, line[0])
                    img = Img.Intelligent_cut(img)
                    self.img_mod.append(img)
                for img in self.img_mod:
                    cv2.imwrite(f'data/mod/img/{num}.jpg', img)
                    num += 1
        elif self.com_out_ans_bool == 4:
            self.com_out_optimize = '需要调整图像设置'

    def thread_hash(self,num, line):
        try:
            img1 = self.img_mod[num]
            img2 = Img.cut(self.midimg, line[0])
            img3 = Img.Intelligent_cut(img2)
            height1, width1 = img1.shape[:2]
            img3 = cv2.resize(img3, (width1, height1))
            hash1 = Img.aHash(img1)
            hash2 = Img.aHash(img3)
            real = Img.cmpHash(hash1, hash2)
            self.hashmark[num] = int(real * 100)
            self.hashans[num] = 0
        except:
            self.hashmark[num] = 0
            self.hashans[num] = 0

    def match(self):
        """
        模板匹配
        """
        t0 = time.time()
        self.midimg = self.img_processing()
        result = self.ppocr.ocr(self.midimg, det=True, rec=True, cls=False)
        text, real = self.data_processing(result)
        self.save = str(time.strftime("%H:%M:%S", time.localtime()))  # 历史记录
        z = self.testtarget.split()  # 识别对比参数
        outplc = '';
        outpc = '模板匹配模式';
        outlen = 0;
        result = {}
        if len(text) == len(z):
            for line in text:
                outplc += f'{line[1]} '
                outlen += len(line[1])
        else:
            if len(text) < len(z):
                for line in text:
                    outplc += f'{line[1]} '
                    outlen += len(line[1])
            if len(text) > len(z):
                text2 = text.copy()
                for line in z:
                    i = 0
                    stri = ""
                    numi = 0
                    num = 0
                    for line2 in text2:
                        s = difflib.SequenceMatcher(None, line, line2[1]).ratio()
                        if s >= i:
                            i = s
                            stri = line2
                            num = numi
                        numi += 1
                    if i == 1.0:
                        del text2[num]
                    else:
                        for x in line:
                            result[x] = result.get(x, 0) + 1
                        for x in stri[1]:
                            result[x] = result.get(x, 0) - 1
                        del text2[num]
                for list in text2:
                    l = list[1]
                    m = 0
                    for list2 in text:
                        if l == list2[1]:
                            del text[m]
                        m += 1
                for list in text:
                    outplc += f'{list[1]} '
                    outlen += len(list[1])
        num = 0
        minmark = 100
        mark=0
        for line in text:
            self.hashans[num] = 1
            self.time_main = threading.Thread(target=self.thread_hash(num ,line))  # 创建对比线程
            self.time_main.start()
            num += 1

        while self.hashans != [0, 0, 0, 0, 0, 0, 0]:
            time.sleep(0.001)
        for i in range(num+1):
            if mark ==0 :
                mark = self.hashmark[i]
            else:
                mark+=self.hashmark[i]

        mark=int(mark/num)
        minmark = min(self.hashmark[0:num])
        self.com_out_mark = mark
        if minmark >= self.maxmark:
            self.com_out_ans_bool = 1
            self.com_out_optimize = '无'
            self.save += f'   合格  检测时间:{(time.time() - t0) * 1000:.0f}ms  置信度:{self.com_out_mark}%  最低匹配度:{int(minmark)}%\n'
            self.com_out_fullans = outpc
            self.com_out_ans = outplc
            self.com_out_ans_len = outlen
            self.com_out_optimize = '无'
            if self.bad_mark > 0:
                if self.bad_mark < 0.8:
                    self.bad_mark = 0
                else:
                    self.bad_mark *= 0.98
        else:
            self.com_out_ans_bool = 2
            self.com_out_optimize = '无'
            self.save += f'   不良  检测时间:{(time.time() - t0) * 1000:.0f}ms  置信度:{self.com_out_mark}%  最低匹配度:{int(minmark)}%\n'
            self.com_out_fullans = outpc
            self.com_out_ans = outplc
            self.com_out_ans_len = outlen
            self.bad_mark += 1
            if self.badnum[4] == 1:
                self.com_out_optimize = '建议重新匹配模板'
            else:
                self.com_out_optimize = '无'
        boxes = [line[0] for line in text]
        endimg = draw_ocr(self.midimg, boxes)
        showend = QtGui.QImage(endimg.data, endimg.shape[1], endimg.shape[0],
                               QtGui.QImage.Format.Format_RGB888)
        self.ui.label_22.setPixmap(QtGui.QPixmap.fromImage(showend))

    def data_processing(self, result):
        """
        # 数据处理
        :param result:
        :return: test,real
        """
        # 提取过滤结果
        text = []  # 结果存放列表
        real = 0.0  # 置信度
        for line in result:
            y = []
            if self.usechar == 1 and self.usecn == 1:
                x = re.sub("[^-年月日/.:0-9]", "", line[1][0])
            elif self.usecn == 1:
                x = re.sub("[^年月日0-9]", "", line[1][0])
            elif self.usechar == 1:
                x = re.sub("[^-/.:0-9]", "", line[1][0])
            else:
                x = re.sub("[^0-9]", "", line[1][0])
            if x != "":
                y.append(line[0])
                y.append(x)
                y.append(line[1][1])
                real += y[2]
                text.append(y)
        if real != 0:
            real /= len(text)
        return text, real

    def img_HSV(self, img):
        """
        彩图HSV自适应归一化
        :param img:
        :return: img
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        channels = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(channels[2], channels[2])
        cv2.merge(channels, hsv)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def img_processing(self):
        """
        图像处理
        :return: img
        """
        if self.img_colour_mod == 1:  # 灰度图
            if self.img_gray_mod == 1:
                # img0 = self.img_HSV(self.show)
                img0 = cv2.cvtColor(self.show, cv2.COLOR_RGB2GRAY)
                min, max = cv2.minMaxLoc(img0)[:2]
                Omin, Omax = 0, 255
                a = float(Omax - Omin) / (max - min)
                b = Omin - a * min
                out = a * img0 + b
                img1 = out.astype(np.uint8)
            elif self.img_gray_mod == 2:
                # img0 = self.img_HSV(self.show)
                img0 = cv2.cvtColor(self.show, cv2.COLOR_RGB2GRAY)
                # 限制对比度的自适应阈值均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img1 = clahe.apply(img0)
            elif self.img_gray_mod == 3:
                # img0 = self.img_HSV(self.show)
                img0 = cv2.cvtColor(self.show, cv2.COLOR_RGB2GRAY)
                img1 = cv2.equalizeHist(img0)
            elif self.img_gray_mod == 0:
                img0 = cv2.cvtColor(self.show, cv2.COLOR_RGB2GRAY)
                img1 = img0
            elif self.img_gray_mod == 4:
                hsv = cv2.cvtColor(self.show, cv2.COLOR_BGR2HSV)
                channels = cv2.split(hsv)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe.apply(channels[2], channels[2])
                cv2.merge(channels, hsv)
                img1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1 = self.blur(img1)
            img9 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        elif self.img_colour_mod == 4:  # 彩图
            img9 = cv2.cvtColor(self.show, cv2.COLOR_BGR2RGB)
            if self.img_colour2_mod == 1:
                img9 = self.img_HSV(img9)
        else:  # 黑白图
            img0 = self.img_HSV(self.show)
            if self.img_black_mod == 1:
                img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
                ret, img1 = cv2.threshold(img0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif self.img_black_mod == 2:
                img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
                img1 = cv2.adaptiveThreshold(img0, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.jdtime,
                                             self.lbtime)
            elif self.img_black_mod == 3:
                img1 = cv2.cvtColor(img0, cv2.COLOR_HSV2RGB)
                img2 = ((img1[:, :, 0] + self.jd2time * 10 >= img1[:, :, 1])
                        * (img1[:, :, 0] + self.jd2time * 10 >= img1[:, :, 2])
                        * (img1[:, :, 2] + self.jd2time * 10 >= img1[:, :, 1])
                        * (img1[:, :, 2] + self.jd2time * 10 >= img1[:, :, 0])
                        * (img1[:, :, 1] + self.jd2time * 10 >= img1[:, :, 2])
                        * (img1[:, :, 1] + self.jd2time * 10 >= img1[:, :, 0])
                        * (img1[:, :, 0] < self.lb2time * 10)
                        * (img1[:, :, 1] < self.lb2time * 10)
                        * (img1[:, :, 2] < self.lb2time * 10))
                img2 = np.invert(img2)
                img2.dtype = 'uint8'
                img1 = img2 * 255
            img1 = self.blur(img1)
            img9 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        return img9

    def ocr(self):
        t0 = time.time()
        # 调用ocr
        self.midimg = self.img_processing()
        # self.midimg = self.show
        result = self.ppocr.ocr(self.midimg, det=True, rec=True, cls=False)

        text, real = self.data_processing(result)

        self.com_out_mark = int(real * 100)  # 输出的置信度
        self.save = str(time.strftime("%H:%M:%S", time.localtime()))  # 历史记录
        # self.com_out_time = int((time.time() - t0) * 1000)  # 检测时间
        z = self.testtarget.split()  # 识别对比参数
        #  结果对比与输出统计
        if self.testtarget == '':
            self.com_out_bad = 0
            if self.com_out_mark >= self.maxmark:
                self.com_out_ans_bool = 1
                self.save += f'   合格  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:0  置信度:{self.com_out_mark}%\n'
                self.com_out_optimize = '未设置检测标准'
            else:
                self.com_out_ans_bool = 2
                self.save += f'   不良  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:0  置信度:{self.com_out_mark}%\n'
                self.com_out_optimize = '置信度异常'
            self.com_out_fullans = ''
            self.com_out_ans = ''
            self.com_out_ans_len = 0
            for line in text:
                self.com_out_fullans += f'{line[1]}\n'
                self.com_out_ans += f'{line[1]} '
                self.com_out_ans_len += len(line[1])
            if self.bad_mark > 0:
                if self.bad_mark < 0.8:
                    self.bad_mark = 0
                else:
                    self.bad_mark *= 0.98
        else:
            outplc = ''
            outpc = ''
            outlen = 0
            znum = 0
            bada = 0
            badb = 0
            result = {}
            if len(text) == len(z):
                for line in text:
                    outplc += f'{line[1]} '
                    outlen += len(line[1])
                    outpc += f'{line[1]}\n'
                    if line[1] != z[znum]:
                        for x in line[1]:
                            result[x] = result.get(x, 0) + 1
                        for x in z[znum]:
                            result[x] = result.get(x, 0) - 1
                    znum += 1
                for x in result:
                    if result[x] < 0:
                        bada -= result[x]
                    if result[x] > 0:
                        badb += result[x]
                if bada < badb: bada = badb
                if self.maxbad >= bada and self.com_out_mark >= self.maxmark:
                    self.com_out_ans_bool = 1
                    self.com_out_bad = bada
                    self.save += f'   合格  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                    self.com_out_fullans = outpc
                    self.com_out_ans = outplc
                    self.com_out_ans_len = outlen
                    if self.bad_mark > 0:
                        if self.bad_mark < 0.8:
                            self.bad_mark = 0
                        else:
                            self.bad_mark *= 0.98
                else:
                    self.com_out_ans_bool = 2
                    self.com_out_bad = bada
                    self.save += f'   不良  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                    self.com_out_fullans = outpc
                    self.com_out_ans = outplc
                    self.com_out_ans_len = outlen
                    self.bad_mark += 1
            else:
                if len(text) < len(z):
                    for line in z:
                        for x in line:
                            result[x] = result.get(x, 0) - 1
                    for line in text:
                        outplc += f'{line[1]} '
                        outlen += len(line[1])
                        outpc += f'{line[1]}\n'
                        for x in line[1]:
                            result[x] = result.get(x, 0) + 1
                    for x in result:
                        if result[x] < 0:
                            bada -= result[x]
                        else:
                            badb += result[x]
                    if bada < badb: bada = badb
                    if self.maxbad >= bada and self.com_out_mark >= self.maxmark:
                        self.com_out_ans_bool = 1
                        self.com_out_bad = bada
                        self.save += f'   合格  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                        self.com_out_fullans = outpc
                        self.com_out_ans = outplc
                        self.com_out_ans_len = outlen
                        if self.bad_mark > 0:
                            if self.bad_mark < 0.8:
                                self.bad_mark = 0
                            else:
                                self.bad_mark *= 0.98
                    else:
                        self.com_out_ans_bool = 2
                        self.com_out_bad = bada
                        self.save += f'   不良  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                        self.com_out_fullans = outpc
                        self.com_out_ans = outplc
                        self.com_out_ans_len = outlen
                        self.bad_mark += 1
                if len(text) > len(z):
                    text2 = text.copy()
                    for line in z:
                        i = 0
                        stri = ""
                        numi = 0
                        num = 0
                        for line2 in text2:
                            s = difflib.SequenceMatcher(None, line, line2[1]).ratio()
                            if s >= i:
                                i = s
                                stri = line2
                                num = numi
                            numi += 1
                        if i == 1.0:
                            del text2[num]
                        else:
                            for x in line:
                                result[x] = result.get(x, 0) + 1
                            for x in stri[1]:
                                result[x] = result.get(x, 0) - 1
                            del text2[num]
                    for list in text2:
                        l = list[1]
                        m = 0
                        for list2 in text:
                            if l == list2[1]:
                                del text[m]
                            m += 1
                    mark = 0
                    for list in text:
                        outplc += f'{list[1]} '
                        outlen += len(list[1])
                        outpc += f'{list[1]}\n'
                        mark += list[2]
                    mark = mark / len(text)
                    for x in result:
                        if result[x] < 0:
                            bada -= result[x]
                        if result[x] > 0:
                            badb += result[x]
                    self.com_out_mark = int(mark * 100)  # 输出的置信度
                    if bada < badb: bada = badb
                    if self.maxbad >= bada and self.com_out_mark >= self.maxmark:
                        self.com_out_ans_bool = 1
                        self.com_out_bad = bada
                        self.save += f'   合格  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                        self.com_out_fullans = outpc
                        self.com_out_ans = outplc
                        self.com_out_ans_len = outlen
                        if self.bad_mark > 0:
                            if self.bad_mark < 0.8:
                                self.bad_mark = 0
                            else:
                                self.bad_mark *= 0.98
                    else:
                        self.com_out_ans_bool = 2
                        self.com_out_bad = bada
                        self.save += f'   不良  检测时间:{(time.time() - t0) * 1000:.0f}ms  不良字符:{bada}  置信度:{self.com_out_mark}%\n'
                        self.com_out_fullans = outpc
                        self.com_out_ans = outplc
                        self.com_out_ans_len = outlen
                        self.bad_mark += 1
        if self.badnum[4] == 1:
            if self.com_out_ans_bool == 2:
                if self.com_out_bad == 0:
                    self.com_out_optimize = '检查打码设备'
                else:
                    self.com_out_optimize = '检查打码或设置'
        else:
            if self.com_out_ans_bool == 1:
                if self.com_out_bad > 0:
                    self.com_out_optimize = '建议减少特殊符号'
                else:
                    self.com_out_optimize = '无'
            else:
                if self.com_out_bad == 0 and self.com_out_mark >= self.maxmark:
                    self.com_out_optimize = '无'
                else:
                    self.com_out_optimize = '建议检查打码/设置'
        boxes = [line[0] for line in text]
        endimg = draw_ocr(self.midimg, boxes)
        showend = QtGui.QImage(endimg.data, endimg.shape[1], endimg.shape[0],
                               QtGui.QImage.Format.Format_RGB888)
        self.ui.label_22.setPixmap(QtGui.QPixmap.fromImage(showend))

    # 输出页面
    def tab_1(self):
        if self.need_save_bool:
            self.scvechange()
        self.ui.tabWidget_2.setCurrentIndex(0)

    # 检测设置
    def tab_2(self):
        self.ui.tabWidget_2.setCurrentIndex(1)

    def freshen_interface(self):
        """
        输出当前检测结果
        """
        self.ui.label_ans_time.setText(str(self.com_out_time) + 'ms')
        if self.com_out_time > 60:
            self.ui.label_ans_time.setStyleSheet('color: rgb(255, 100, 0);')
        else:
            self.ui.label_ans_time.setStyleSheet('')
        self.ui.label_ans_optimize.setText(self.com_out_optimize)
        if self.com_out_optimize == '无':
            self.ui.label_ans_optimize.setStyleSheet('')
        elif re.sub("[^建议]", "", self.com_out_optimize) == '建议':
            self.ui.label_ans_optimize.setStyleSheet('color: rgb(255, 100, 0);')
        else:
            self.ui.label_ans_optimize.setStyleSheet('color: rgb(255, 0, 0);')
        if self.com_out_ans_bool == 1:
            self.ui.label_ans_2.setText('合格')
            self.ui.label_ans.setText('合格')
            self.ui.label_ans.setStyleSheet('color: rgb(0, 200, 0);')
            self.ui.label_ans_2.setStyleSheet('')
            self.ui.label_ans_bad_num.setText(str(self.com_out_bad))
            if self.com_out_bad == 0:
                self.ui.label_ans_bad_num.setStyleSheet('')
            else:
                self.ui.label_ans_bad_num.setStyleSheet('color: rgb(255, 100, 0);')
        elif self.com_out_ans_bool == 2:
            self.ui.label_ans_2.setText('不良')
            self.ui.label_ans.setText('不良')
            self.ui.label_ans.setStyleSheet('color: rgb(255, 0, 0);')
            self.ui.label_ans_2.setStyleSheet('color: rgb(255, 0, 0);')
            self.ui.label_ans_bad_num.setText(str(self.com_out_bad))
            if self.com_out_bad == 0:
                self.ui.label_ans_bad_num.setStyleSheet('')
            else:
                self.ui.label_ans_bad_num.setStyleSheet('color: rgb(255, 0, 0);')
        elif self.com_out_ans_bool == 3:
            self.ui.label_ans_2.setText('学习成功')
            self.ui.label_ans.setText('学习成功')
            self.ui.label_ans.setStyleSheet('color: rgb(0, 200, 0);')
            self.ui.label_ans_2.setStyleSheet('')
            self.ui.label_ans_bad_num.setText(str(self.com_out_bad))
            if self.com_out_bad == 0:
                self.ui.label_ans_bad_num.setStyleSheet('')
            else:
                self.ui.label_ans_bad_num.setStyleSheet('color: rgb(255, 100, 0);')
        elif self.com_out_ans_bool == 4:
            self.ui.label_ans_2.setText('学习失败')
            self.ui.label_ans.setText('学习失败')
            self.ui.label_ans.setStyleSheet('color: rgb(255, 0, 0);')
            self.ui.label_ans_2.setStyleSheet('color: rgb(255, 0, 0);')
            self.ui.label_ans_bad_num.setText(str(self.com_out_bad))
            if self.com_out_bad == 0:
                self.ui.label_ans_bad_num.setStyleSheet('')
            else:
                self.ui.label_ans_bad_num.setStyleSheet('color: rgb(255, 0, 0);')
        self.ui.label_ans_reliability.setText(str(self.com_out_mark) + '%')
        if self.com_out_mark < 80:
            self.ui.label_ans_reliability.setStyleSheet('color: rgb(255, 100, 0);')
        else:
            self.ui.label_ans_reliability.setStyleSheet('')
        self.ui.label_ans_str.setText(self.com_out_fullans)
        if len(self.historylist) < 15:
            self.historylist.append(self.save)
        else:
            del (self.historylist[0])
            self.historylist.append(self.save)
        s = ""
        for i in self.historylist:
            s += f'{i}'
        self.ui.label_ans_his.setPlainText(s)

    def scvechange(self):
        """
        保存设置
        """
        if self.need_learn:
            bool_learn = 1
        else:
            bool_learn = 0
        f = open('data/data', 'w')
        a = f'{self.img_colour_mod}\n{self.img_gray_mod}\n{self.img_black_mod}\n{self.img_colour2_mod}\n{self.img_blus_mod}\n{self.maxbad}' \
            f'\n{self.maxmark}\n{self.opentime}\n{self.closetime}\n{self.jdtime}\n{self.lbtime}\n{self.jd2time}\n{self.lb2time}\n{self.usecn}' \
            f'\n{self.usechar}\n{bool_learn}'
        f.write(a)
        l = open('data/data2', 'w')
        a = self.testtarget
        l.write(a)
        self.need_save_bool = False

    def startset(self):
        """
        初始化设置参数与激活
        """
        f = open('data/data', 'r')
        lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
        self.img_colour_mod = int(lines[0].strip('\n'))
        self.img_gray_mod = int(lines[1].strip('\n'))
        self.img_black_mod = int(lines[2].strip('\n'))
        self.img_colour2_mod = int(lines[3].strip('\n'))
        self.img_blus_mod = lines[4].strip('\n')
        self.maxbad = int(lines[5].strip('\n'))
        self.maxmark = int(lines[6].strip('\n'))
        self.opentime = int(lines[7].strip('\n'))
        self.ui.label_open.setText(str(self.opentime))
        self.closetime = int(lines[8].strip('\n'))
        self.ui.label_close.setText(str(self.closetime))
        self.jdtime = int(lines[9].strip('\n'))
        self.ui.label_jd.setText(str(self.jdtime))
        self.lbtime = int(lines[10].strip('\n'))
        self.ui.label_lb.setText(str(self.lbtime))
        self.jd2time = int(lines[11].strip('\n'))
        self.ui.label_jd_2.setText(str(self.jd2time))
        self.lb2time = int(lines[12].strip('\n'))
        self.ui.label_lb_2.setText(str(self.lb2time))
        self.usecn = int(lines[13].strip('\n'))
        if self.usecn == 1:
            self.ui.Buttoncn.setText('禁用\n中文')
        else:
            self.ui.Buttoncn.setText('使用\n中文')
        self.usechar = int(lines[14].strip('\n'))
        if self.usechar == 1:
            self.ui.Buttoncn_2.setText('禁用\n符号')
        else:
            self.ui.Buttoncn_2.setText('使用\n符号')
        bool_learn = int(lines[15].strip('\n'))
        if bool_learn == 1:
            self.need_learn = True
        if self.img_colour_mod == 1:
            self.ui.radioButtongray.setChecked(True)
        elif self.img_colour_mod == 2:
            self.ui.radioButtonblack.setChecked(True)
        elif self.img_colour_mod == 3:
            self.ui.radioButtonmould.setChecked(True)
        elif self.img_colour_mod == 4:
            self.ui.radioButtoncolor.setChecked(True)
        if self.img_gray_mod == 1:
            self.ui.radioButton_his.setChecked(True)
        elif self.img_gray_mod == 2:
            self.ui.radioButton_hisauto.setChecked(True)
        elif self.img_gray_mod == 3:
            self.ui.radioButton_hisall.setChecked(True)
        elif self.img_gray_mod == 4:
            self.ui.radioButton_hsv.setChecked(True)
        if self.img_black_mod == 1:
            self.ui.radioButton_otus.setChecked(True)
            self.ui.widget_7.hide()
            self.ui.widget_8.hide()
        elif self.img_black_mod == 2:
            self.ui.radioButton_mean.setChecked(True)
            self.ui.widget_7.show()
            self.ui.widget_8.hide()
        elif self.img_black_mod == 3:
            self.ui.radioButton_200.setChecked(True)
            self.ui.widget_8.show()
            self.ui.widget_7.hide()
        if re.sub("[^1]", "", self.img_blus_mod) != "":
            self.ui.radioButtongauss.setChecked(True)
        if re.sub("[^2]", "", self.img_blus_mod) != "":
            self.ui.radioButtoneven.setChecked(True)
        if re.sub("[^3]", "", self.img_blus_mod) != "":
            self.ui.radioButtonopen.setChecked(True)
        if re.sub("[^4]", "", self.img_blus_mod) != "":
            self.ui.radioButtonclose.setChecked(True)
        if self.img_colour_mod == 3:
            self.ui.pushButtoncleanlearn.show()
        else:
            self.ui.pushButtoncleanlearn.hide()
        if self.img_colour_mod == 1:
            self.ui.widget_3.show()
            self.ui.widget_5.hide()
            self.ui.widget_6.hide()
            if self.img_gray_mod == 0:
                self.ui.widget_4.hide()
            else:
                self.ui.widget_4.show()
        elif self.img_colour_mod == 4:
            self.ui.widget_6.show()
            self.ui.widget_5.hide()
            self.ui.widget_3.hide()
        else:
            self.ui.widget_4.show()
            self.ui.widget_5.show()
            self.ui.widget_6.hide()
            self.ui.widget_3.hide()
        if self.img_colour2_mod == 1:
            self.ui.radioButton_colorhis.setChecked(True)
        self.ui.label_bad_4.setText(str(self.maxmark))
        self.ui.label_bad.setText(str(self.maxbad))

        l = open('data/data2', encoding="gb2312", mode='r')
        file = l.read()
        self.testtarget = file
        self.ui.inin.setPlainText(self.testtarget)

        for num in range(len(self.testtarget.split())):
            i = cv2.imread(f'data/mod/img/{num}.jpg')
            self.img_mod.append(i)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    app.exec()
