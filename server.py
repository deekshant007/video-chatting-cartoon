import pickle
import socket
import struct
import cv2
import numpy as np

def color_quantization(img, k):
  data = np.float32(img).reshape((-1, 3))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
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
    img_1 = color_quantization(img, 7)

    #Step 5
    blurred = cv2.medianBlur(img_1, 3)

    #Step 6
    cartoon_1 = cv2.bitwise_and(blurred, blurred, mask=edges)

    return cartoon_1



def simpleRec():
    video = cv2.VideoCapture(0)
   # video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    while True:
        try:
            ret, frame = video.read()
            frame = CartoonFilter(frame)
            cv2.imshow("Frame Server", frame)
            if cv2.waitKey(1) == 13:
                print("you closed")
                cv2.destroyAllWindows()
                break
        except:
            print("Frame Rendering Failed")
            break

    video.release()



def Main():
    socket_address = ('127.0.0.1', 1234)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    s.bind(socket_address)
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    data = b''
    payload_size = struct.calcsize("L")

    while True:
        try:
            ret, ser = video.read()
            ser = CartoonFilter(ser)

            while len(data) < payload_size:
                data += conn.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data)
            frame = cv2.resize(frame, (600, 600))
            server_frame = cv2.resize(ser, (200, 200))
            frame[400:, 400:] = server_frame

            cv2.imshow('Server Side', frame)
            cv2.waitKey(int((1/30)*1000))

            print("Server: Starting server Stream")
            serverdata = pickle.dumps(ser)
            message_size = struct.pack("L", len(serverdata))
            conn.sendall(message_size + serverdata)
            print("Server: Data Sent to client Successfully")

            if cv2.waitKey(1) == 13:
                print("you closed")
                conn.shutdown(2)
                conn.close()
                cv2.destroyAllWindows()
                break
        except:
            cv2.destroyAllWindows()
            print("other person closed")
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Main()
    #simpleRec()