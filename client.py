import cv2
import socket
import pickle
import struct
import threading
import numpy as np

def color_quantization(img, k):
# Defining input data for clustering
  data = np.float32(img).reshape((-1, 3))
# Defining criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# Applying cv2.kmeans function
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result


def CartoonFilter(img):

    #Step 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_1 = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)

    #Step 2
    color = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=200)

    #Step 3
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    #Step 4
    img_1 = color_quantization(cartoon, 7)

    #Step 5
    blurred = cv2.medianBlur(img_1, 3)

    #Step 6
    cartoon_1 = cv2.bitwise_and(blurred, blurred, mask=edges)

    return cartoon_1

def Main():
    video = cv2.VideoCapture(1)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('127.0.0.1', 1234))
    data = b''
    payload_size = struct.calcsize("L")  # unsigned long integer


    while True:
        try:
            print("Sending Stream to Server Now")
            ret, frame = video.read()
            frame = CartoonFilter(frame)
            print("Frame successfully read by client")
            clientdata = pickle.dumps(frame)
            message_size = struct.pack("L", len(clientdata))
            clientsocket.sendall(message_size + clientdata)
            print("Client: Data Sent to server Successfully")


            print("Client: Receiving Sream from the Server Now")
            while len(data) < payload_size:
                data += clientsocket.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += clientsocket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            server_frame = pickle.loads(frame_data)
            server_frame = cv2.resize(server_frame, (600, 600))
            client_frame = cv2.resize(frame, (200, 200))
            server_frame[400:, 400:] = client_frame

            cv2.imshow('Client Side', server_frame)
            cv2.waitKey(int((1 / 30) * 1000))

            if cv2.waitKey(1) == 27:
                print("you closed")
                clientsocket.shutdown(2)
                clientsocket.close()
                cv2.destroyAllWindows()
                break
        except:
            cv2.destroyAllWindows()
            print("other person closed")
            break

    video.release()





#New Program

# import socket
# import threading
# import cv2
# import pickle
# import struct
#
# def Program():
#     s = socket.socket()
#     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     s.connect(('192.168.99.1', 3333))
#
#     cap = cv2.VideoCapture(1)
#
#     def recv():
#         size_of_msg = struct.calcsize("L")
#         data = b''
#         while True:
#
#             while len(data) < size_of_msg:
#                 data += s.recv(1024)
#             true_msg_size = data[:size_of_msg]
#             data = data[size_of_msg:]
#             msg_size = struct.unpack("L", true_msg_size)[0]
#             while len(data) < msg_size:
#                 data += s.recv(4096)
#             image_data = data[:msg_size]
#             data = data[msg_size:]
#             image = pickle.loads(image_data)
#             ret1, live1 = cap.read()
#             frame1 = cv2.resize(live1, (240, 240))
#             image[240:, 400:] = frame1
#
#             cv2.imshow('in sender file of 1 camera', image)
#             if cv2.waitKey(1) == 13:
#                 break
#         cv2.destroyAllWindows()
#
#     t1 = threading.Thread(target=recv)
#     t1.start()
#
#     live_img = cv2.imread('live-streaming.png')
#     frame = cv2.resize(live_img, (620, 480))
#     while True:
#         ret, photo = cap.read()
#         image = pickle.dumps(photo)
#         img_size = struct.pack("L", len(image))
#         s.sendall(img_size + image)
#
#     cv2.destroyAllWindows()
#     cap.release()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Main()
    #Program()

# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
