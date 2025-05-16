import cv2
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, request, Response, send_from_directory, session, redirect, url_for
from flask_session import Session
import threading
import time
from datetime import datetime
import os
import json

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Video yükleme için klasör
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Profil resmi için klasör
PROFILE_FOLDER = 'static/profile'
app.config['PROFILE_FOLDER'] = PROFILE_FOLDER
os.makedirs(PROFILE_FOLDER, exist_ok=True)

# Tespit görüntüleri için klasör
DETECTION_FOLDER = 'static/detections'
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Admin bilgilerini saklamak için JSON dosyası
ADMIN_INFO_FILE = 'admin_info.json'

# Varsayılan admin bilgileri
default_admin_info = {
    "name": "Admin Adı",
    "profile_picture": "default_profile.jpg"
}

# YOLOv8 modelini yükle (indirdiğin best.pt ile)
model = YOLO('best.pt')  # İndirdiğin modeli buraya koy

# Yangın ve duman sınıfları (modelin data.yaml dosyasından doğrula)
FIRE_CLASSES = ['fire', 'smoke']  # GitHub deposundaki sınıf isimleriyle eşleşmeli

# Kamera bilgileri
cameras = [
    {"id": 1, "path": None, "region": "Bölge 1", "cap": None, "status": "Kapalı", "thread": None},
    {"id": 2, "path": None, "region": "Bölge 2", "cap": None, "status": "Kapalı", "thread": None},
    {"id": 3, "path": None, "region": "Bölge 3", "cap": None, "status": "Kapalı", "thread": None},
    {"id": 4, "path": None, "region": "Bölge 4", "cap": None, "status": "Kapalı", "thread": None},
]

# Bildirimler
notifications = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

# Admin bilgilerini yükle
def load_admin_info():
    if not os.path.exists(ADMIN_INFO_FILE):
        with open(ADMIN_INFO_FILE, 'w') as f:
            json.dump(default_admin_info, f)
    with open(ADMIN_INFO_FILE, 'r') as f:
        return json.load(f)

# Admin bilgilerini kaydet
def save_admin_info(admin_info):
    with open(ADMIN_INFO_FILE, 'w') as f:
        json.dump(admin_info, f)

# Kullanıcı doğrulama
def is_valid_user(username, password):
    users = {
        "Ruslan": "ruslan123",
        "Nurbek": "nurbek123",
        "Samat": "samat123"
    }
    return users.get(username) == password

# Video işleme ve yangın/duman tespiti
def process_video(camera):
    cap = camera["cap"]
    if not cap or not cap.isOpened():
        camera["status"] = "Hata"
        print(f"{camera['region']}: Video açılamadı.")
        return

    frame_skip = 15  # Performans için her 15 karede bir analiz
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"{camera['region']}: Video sona erdi veya okunamadı, başa dönülüyor...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Video bittiğinde başa dön
            frame_count = 0
            time.sleep(0.1)  # Kısa bir bekleme ekle
            continue

        frame_count += 1
        camera["frame"] = frame.copy()  # Her kareyi güncelle
        print(f"{camera['region']}: Kare güncellendi, Frame Sayısı: {frame_count}")

        if frame_count % frame_skip == 0:
            try:
                results = model.predict(frame, conf=0.5)  # Güven eşiğini 0.5 olarak ayarla
                fire_detected = False
                smoke_detected = False
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Kutu koordinatları
                    confidences = result.boxes.conf.cpu().numpy()  # Güven skorları
                    classes = result.boxes.cls.cpu().numpy()  # Sınıf indeksleri
                    names = result.names  # Modelin sınıf isimleri

                    for box, conf, cls in zip(boxes, confidences, classes):
                        class_name = names[int(cls)]
                        if class_name in FIRE_CLASSES:
                            x1, y1, x2, y2 = map(int, box)
                            color = (0, 0, 255) if class_name == 'fire' else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            if class_name == 'fire':
                                fire_detected = True
                            elif class_name == 'smoke':
                                smoke_detected = True

                if fire_detected or smoke_detected:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    message = "Ateş tespit edildi!" if fire_detected else "Duman tespit edildi!" if smoke_detected else "Tespit edildi!"
                    image_path = os.path.join(DETECTION_FOLDER, f"detected_{camera['id']}_{'fire' if fire_detected else 'smoke'}_{timestamp.replace(':', '-')}.jpg")
                    cv2.imwrite(image_path, frame)
                    notifications.append({
                        "region": camera["region"],
                        "message": message,
                        "timestamp": timestamp,
                        "image_path": image_path
                    })
                    print(f"{camera['region']}: {message} Bildirim eklendi.")
            except Exception as e:
                print(f"{camera['region']}: YOLOv8 hata: {str(e)}")

        if camera["status"] != "Çalışıyor":
            break

    cap.release()
    camera["cap"] = None
    print(f"{camera['region']}: Video kapatma işlemi tamamlandı.")

# Canlı yayın için kareleri oluştur
def generate_frames(camera):
    try:
        while True:
            if camera["status"] != "Çalışıyor":
                print(f"{camera['region']}: Durum Çalışıyor değil, akış durduruldu.")
                time.sleep(0.1)
                continue

            if "frame" not in camera or camera["frame"] is None:
                print(f"{camera['region']}: Kare bulunamadı, bekleniyor...")
                time.sleep(0.1)
                continue

            frame = camera["frame"]
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print(f"{camera['region']}: Kare kodlama hatası")
                time.sleep(0.1)
                continue
            frame = buffer.tobytes()
            print(f"{camera['region']}: Kare gönderildi")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"{camera['region']}: Akış hatası: {str(e)}")

# Giriş ekranı
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if is_valid_user(username, password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Kullanıcı adı veya şifre yanlış!")
    return render_template('login.html')

# Çıkış
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Ana sayfa (dashboard)
@app.route('/')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    admin_info = load_admin_info()
    return render_template('index.html', cameras=cameras, notifications=notifications, admin_info=admin_info, username=session['username'])

# Admin bilgilerini güncelle
@app.route('/update_admin', methods=['POST'])
def update_admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    admin_info = load_admin_info()

    if 'admin_name' in request.form:
        admin_info['name'] = request.form['admin_name']

    if 'profile_picture' in request.files:
        file = request.files['profile_picture']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['PROFILE_FOLDER'], filename)
            file.save(file_path)
            admin_info['profile_picture'] = filename

    save_admin_info(admin_info)
    return jsonify({"success": "Admin bilgileri güncellendi"}), 200

# Video yükleme
@app.route('/upload/<int:camera_id>', methods=['POST'])
def upload_video(camera_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    camera = next((cam for cam in cameras if cam["id"] == camera_id), None)
    if not camera:
        return jsonify({"error": "Kamera bulunamadı"}), 404

    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Dosya seçilmedi"}), 400

    if file and allowed_file(file.filename):
        filename = f"camera_{camera_id}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Mevcut kaynakları temizle
        if camera["cap"] is not None:
            camera["cap"].release()
            camera["cap"] = None
        if camera["thread"] is not None:
            camera["status"] = "Kapalı"
            camera["thread"].join()
            camera["thread"] = None

        camera["path"] = file_path
        camera["cap"] = cv2.VideoCapture(file_path)
        if not camera["cap"].isOpened():
            camera["status"] = "Hata"
            return jsonify({"error": "Video açılamadı"}), 400

        camera["status"] = "Çalışıyor"
        camera["thread"] = threading.Thread(target=process_video, args=(camera,))
        camera["thread"].start()
        print(f"{camera['region']}: Video yüklendi - Path: {file_path}")
        return jsonify({"success": "Video yüklendi", "path": file_path}), 200

    return jsonify({"error": "Geçersiz dosya formatı"}), 400

# Video silme
@app.route('/delete/<int:camera_id>', methods=['POST'])
def delete_video(camera_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    camera = next((cam for cam in cameras if cam["id"] == camera_id), None)
    if not camera:
        return jsonify({"error": "Kamera bulunamadı"}), 404

    if camera["path"]:
        # Önce thread’i ve video kaynağını kapat
        if camera["thread"] is not None:
            camera["status"] = "Kapalı"
            camera["thread"].join(timeout=5)  # Thread’in kapanmasını bekle
            if camera["thread"].is_alive():
                print(f"{camera['region']}: Thread kapanmadı, zorla durduruluyor...")
            camera["thread"] = None

        if camera["cap"] is not None:
            camera["cap"].release()
            camera["cap"] = None
            print(f"{camera['region']}: VideoCapture serbest bırakıldı")

        # Dosyayı sil
        try:
            if os.path.exists(camera["path"]):
                os.remove(camera["path"])
                print(f"{camera['region']}: Dosya başarıyla silindi: {camera['path']}")
            else:
                print(f"{camera['region']}: Dosya bulunamadı: {camera['path']}")
        except Exception as e:
            print(f"{camera['region']}: Dosya silme hatası: {str(e)}")
            return jsonify({"error": f"Dosya silme hatası: {str(e)}"}), 500

        camera["path"] = None
        camera["status"] = "Kapalı"
        print(f"{camera['region']}: Video silindi ve durum güncellendi.")
        return jsonify({"success": "Video silindi"}), 200

    return jsonify({"error": "Video bulunamadı"}), 400

# Canlı yayın
@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    camera = next((cam for cam in cameras if cam["id"] == camera_id), None)
    if not camera:
        return jsonify({"error": "Kamera bulunamadı"}), 404
    try:
        return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Video feed hatası ({camera_id}): {str(e)}")
        return jsonify({"error": "Akış başlatılamadı"}), 500

# Bildirimler
@app.route('/notifications')
def get_notifications():
    if 'username' not in session:
        return redirect(url_for('login'))
    return jsonify(notifications)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)