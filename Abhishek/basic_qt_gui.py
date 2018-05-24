import sys
from PyQt4 import QtGui, QtCore

class Window(QtGui.QMainWindow):

	def __init__(self):
		super(Window, self).__init__()
		geometry = QtGui.QDesktopWidget().availableGeometry()
		# print( geometry)
		l=geometry.width()
		h=geometry.height()
		print(l,h)
		self.setGeometry(geometry)
		# self.showFullScreen()
		self.setWindowTitle("PyQt data_gui")
		self.home(l,h)
		self.buttons = []

	def mousePressEvent(self, QMouseEvent):
		print (QMouseEvent.pos())

	def mouseReleaseEvent(self, QMouseEvent):
		cursor = QtGui.QCursor()
		print (cursor.pos())

	def mouseReleaseEvent(self, QMouseEvent):
		print('(', QMouseEvent.x(), ', ', QMouseEvent.y(), ')')

	def home(self, width, height):
		l=width
		h=height
		j=0
		k=0
		data = []
		print(l, h)
		for i in range(((l-l%50)*(h-h%50)//2500)):
			btn = QtGui.QPushButton("T" + str(i), self)
			# self.buttons.append(btn)
			btn.clicked.connect(self.handleButton(i,j,data))
			btn.resize(50, 50)

			if ((i)%(int(l/50))==0 and i>0):
				k+=(l-int(l%50))
				j+=1

			btn.move(i*50-k, j*50)

		self.show()
	def handleButton(self,i,j,data):
		def calluser():
			data.append((i,j))
			print (data)
		return calluser


def main():
	l=0
	h=0
	app = QtGui.QApplication(sys.argv)
	GUI = Window()
	sys.exit(app.exec_())

main()