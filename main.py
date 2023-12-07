import numpy as np
import os
import cv2
from imageai.Detection import ObjectDetection
from colors import COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_BLUE
import coordinates_generator
import yaml
from shapely.geometry import Polygon

import drawing_utils


#def get_coords(event, x, y, flags, param):
#    global get_x, get_y
#    if event == cv2.EVENT_LBUTTONDOWN:
#        print(x, y)
#        get_x, get_y = (x, y)


#def max_diagonal(vertices):
#    max_distance = 0
#    rect = list()
#    for i in range(len(vertices)):
#        for j in range(i + 1, len(vertices)):
#            distance = np.sqrt((vertices[j][0] - vertices[i][0]) ** 2 + (vertices[j][1] - vertices[i][1]) ** 2)
#            if distance > max_distance:
#                max_distance = distance
#                rect = np.array([(vertices[j][0], vertices[j][1]), (vertices[i][0], vertices[i][1])])
#    return rect


#def area(dots):
#    return 0.5 * abs(
#        (dots[0][0] * dots[1][1] + dots[1][0] * dots[2][1] + dots[2][0] * dots[3][1] + dots[3][0] * dots[0][1]) - (
#                    dots[0][1] * dots[1][0] + dots[1][1] * dots[2][0] + dots[2][1] * dots[3][0] + dots[3][1] * dots[0][
#                0]))


def occupation(_spot, _cars):
    all_cars = []
    j = 0
    IoU = 0.
    for car in _cars:
        if len(car['box_points']) == 4:
            all_cars.append(car['box_points'])
            all_cars[j].append(car['box_points'][0])
            all_cars[j].append(car['box_points'][3])
            all_cars[j].append(car['box_points'][2])
            all_cars[j].append(car['box_points'][1])
            all_cars[j] = np.array_split(all_cars[j], 4)
        else:
            all_cars.append(car['box_points'])
            all_cars[j] = np.array_split(all_cars[j], 4)

        car_poly = Polygon(all_cars[j])
        spot_poly = Polygon(_spot)
        if spot_poly.intersects(car_poly):
            intersec = car_poly.buffer(0).intersection(spot_poly.buffer(0))
            intersec_area = intersec.area
            total_area = car_poly.area + spot_poly.area
            if IoU < (intersec_area / total_area):
                IoU = intersec_area / total_area
        j += 1

    return IoU


# подключение камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 25)

# подготовка камеры
for i in range(30):
    cap.read()

ret, img = cap.read()

# pts = np.array(range(6) ,'f')
# i = 0
# for i in range(0, 6, 2):
# cv2.imshow('image', img)
# cv2.setMouseCallback("image", get_coords)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# pts[i] = get_x
# pts[i + 1] = get_y

# i_pts = np.float32([[pts[0], pts[0]], [pts[1], pts[1]], [pts[2], pts[2]]])
# o_pts = np.float32([[0., 0.], [14.5, 0], [13.5, 18.]])


# affine_mat = cv2.getAffineTransform(i_pts, o_pts)

#img = cv2.imread('img9.jpg')

cv2.imwrite('first.jpg', img)

with open("coords.yml", "w+") as points:
    generator = coordinates_generator.CoordinatesGenerator("first.jpg", points, COLOR_RED)
    generator.generate()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolov3.pt")
detector.loadModel()
custom_obj = detector.CustomObjects(car=True, motorbike=True, bus=True, truck=True)

i = 0
while True:
    execution_path = os.getcwd()
#    ret, img = cap.read()
#    cv2.imshow(f"cam{i}", img)
#    cv2.imwrite(f"img{i}.jpg", img)
    # detected = detection()
    # affin = cv2.warpAffine(img, affine_mat, (img.shape[1], img.shape[0]))
    # img_affin = cv2.hconcat([img, affin])
    # cv2.imwrite(f'img_a{i}.jpg', affin)
    # cv2.imshow('affin', affin)
    detections = []
    with open(os.path.join("coords.yml"), "r") as data:
        points = yaml.safe_load(data)
        detections = detector.detectObjectsFromImage(custom_objects=custom_obj, input_image=img,
                                                     output_image_path=os.path.join("detected_img.jpg"),
                                                     minimum_percentage_probability=40)
        img_detected = cv2.imread("detected_img.jpg")
#        cv2.imshow("detected", img_detected)
        states = []

        for slots in points:
            is_ok = occupation(slots.get('coordinates'), detections)
            if is_ok >= 0.2:
                states.append(True)
            else:
                states.append(False)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for index, slots in enumerate(points):
            if states[index]:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                color = COLOR_BLUE
                drawing_utils.draw_contours(img, np.array(slots.get('coordinates'), np.int32), str(slots['id']),
                                            COLOR_WHITE, color)
            else:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                color = COLOR_GREEN
                drawing_utils.draw_contours(img, np.array(slots.get('coordinates'), np.int32), str(slots['id']),
                                            COLOR_WHITE, color)

            cv2.imshow("res", img)




    if cv2.waitKey(10000) == 27:
        break
    cv2.destroyAllWindows()
    i += 1

cap.release()
cv2.destroyAllWindows()
