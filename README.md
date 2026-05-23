A basic browser-based pipeline project for American Sign Language (ASL) fingerspelling recognition.
Custom training data can be collected through webcam, based on which a neural network using TensorFlow can be trained and then used for real-time inference in the browser.

FEAT:
1. Data Collector: Web-based, can record 21-point hand landmarks(<from Mediapipe) for all 26 letters.
2. Robust: Scale-normalised features with structural distances (finger curl, fist tightness) and palm orientation vectors to resolve common ASL confusions (E vs M, H vs G, P vs Q, R vs K)
3. Tensorflow model with data augmentation, early stopping and custom export to TensorFlow.js format
4. Runs entirely on browser with no backend required.
