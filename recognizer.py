import cv2
import numpy as np

# set deteksi wajah
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();

# baca trainning data
rec.read('recognizer/trainningData.yml')

#
id=0

# pilih jenis font
font=cv2.FONT_HERSHEY_SIMPLEX

while(True):
	
	# baca kamera
	ret,img=cam.read();

	# ubah gambar ke gray
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# deteksi scale wajah
	faces=faceDetect.detectMultiScale(gray,1.3,5);

	# nama orang
	name_str = 'Unknown Person'

	# multiple
	multiple_person = []

	# iterasi hasil deteksi fajar untuk prediksi wajah
	for(x,y,w,h) in faces:

		# buat kotak rectangle
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

		# prediksi gambar
		id,conf=rec.predict(gray[y:y+h,x:x+w])

		# periksa prediksi
		if conf < 60 : 
			# seleksi nama berdasarkan id yang sudah diregistrasi
			if(id == 123):
				name_str = 'Fajar'
			elif(id == 124):
				name_str = 'Oki'
			elif(id == 125):
				name_str = 'Ridwansyah'
		else:
			name_str = 'Unknown'

		# menampilkan nama wajah yang di scan 
		# disini akan berisi berbeda-beda rectangle (kotak wajah)
		cv2.putText(img,name_str,(x,y+h-10),font,1,(255, 255, 255));
	
	# menampilkan frame untuk menampilkan recognizer
	# tidak ditaruh didalam iterasi for karena
	# tampilan gambar akan berhenti
	cv2.imshow("~ Multi Deteksi Wajah ~",img);

	print id
	
	if(cv2.waitKey(1)==ord('q')):
		break;
cam.release()
cv2.destroyAllWindows()
