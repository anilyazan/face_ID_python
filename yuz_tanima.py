import cv2

# OpenCV kütüphanesinden eğitilmiş yüz tanıma modelini içe aktarın
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamerayı başlatın (0 yerine kameranın indeksi kullanılabilir, örneğin 1)
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare okuyun
    ret, frame = cap.read()

    # Eğer kare okunamazsa döngüyü sonlandır
    if not ret:
        break

    # Yüzleri tespit etmek için kareyi gri tona dönüştürün
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit etmek için eğitilmiş modeli kullanarak tespit yapın
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Tespit edilen yüzlerin etrafına dikdörtgen çizin
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Tanımlanan yüzlerin olduğu kareyi ekrana gösterin
    cv2.imshow('Face Detection', frame)

    # Q tuşuna basarak çıkış yapın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereyi serbest bırakın
cap.release()
cv2.destroyAllWindows()
