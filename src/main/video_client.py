import numpy as np
import cv2 as cv
import hazelcast

framex = None

def on_message(event):
    try:
        frame_data = event.message

        frame = cv.imdecode(np.frombuffer(frame_data, np.uint8), cv.IMREAD_ANYCOLOR)

        global framex
        framex = frame
    except Exception as e: print(e)

client = hazelcast.HazelcastClient()
topic = client.get_reliable_topic("topic").blocking()
topic.add_listener(on_message)


while True:
    if framex is not None:
        cv.imshow('frame', framex)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()