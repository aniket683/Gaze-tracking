import sys
from PyQt4 import QtGui, QtCore

class mainUI(QtGui.QWidget):
        def __init__(self):
                super(mainUI, self).__init__()
                self.initUI()

        def initUI(self):
                l = QtGui.QDesktopWidget().availableGeometry()
                print (l)

                qbtn = QtGui.QPushButton('Quit')
                qbtn.clicked.connect(QtCore.QCoreApplication.quit)
                qbtn.move(50,50)
                self.button = qbtn
                qbtn.show()


def main():
        app = QtGui.QApplication(sys.argv)
        window = mainUI()
        sys.exit(app.exec_())

if __name__ == '__main__':
        main()