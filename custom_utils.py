import cv2

def filter_detections(results, confidence=0.5):
    confidence_flt = results['confidence'] >= confidence
    class_flt = results['class'] == 0
    return results[confidence_flt&class_flt]


def detect_yolo(model, img):
    results = model(img)
    df = results.pandas().xyxy[0]
    df_filtered = filter_detections(df)
    areas = []
    centers = []
    for id in range(len(df_filtered)):
        xmin, ymin, xmax, ymax, _, _, _ = df_filtered.iloc[id]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        box = ((xmin,ymin), (xmax, ymax))
        area = (xmax-xmin) * (ymax-ymin)
        center = ((xmin+xmax)//2, (ymin+ymax)//2)

        areas.append((area, box))
        centers.append((center, box))
    return areas, centers



def show(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_haarcascade(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_areas = []
    face_centers = []
    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        box = ((x,y), (x+w, x+h))
        face_areas.append((area, box))
        face_centers.append(((cx, cy), box))

    print("face_areas: ", face_areas)
    print("face_centers: ", face_centers)

    if len(face_areas) !=0:
        max_idx = face_areas.index(max(face_areas, key=lambda x:x[0]))
        face_area = [face_areas[max_idx]]
        face_center = [face_centers[max_idx]]
        print("face_area: ", face_area)
        print("face_center: ", face_center)
        return face_area, face_center
    else:
        return [], []
    


def draw_figures(img, centers):
    for center_ in centers:
        print("center_:", center_)
        center, bbox = center_
        start_point = bbox[0]
        end_point = bbox[1]
        img = cv2.rectangle(img, start_point, end_point, (0,255,255), 1)
        img = cv2.circle(img, center, 5, (0,0,255), cv2.FILLED)
    return img
